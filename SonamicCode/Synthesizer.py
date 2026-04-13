"""
Generic audio engine.

AudioState categories:
    1. ONE-SHOT tones  — pre-rendered numpy arrays, played once then discarded.
    2. CONTINUOUS tones — additive sine synthesis from freq/amp arrays.

AudioEngine : Owns the sounddevice stream. One instance for the whole session.
"""

import numpy as np
import sounddevice as sd
import threading


class AudioState:
    """
    central audio state container.
    written by audification classes (main thread).
    read by the audio callback (audio thread).
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.lock = threading.Lock()

        # one-shot storage: list of [samples_array, read_head]
        self._oneshots = []

        # continuous tone storage: parallel arrays
        self._cont_freqs  = None   # (N,) Hz
        self._cont_amps   = None   # (N,) amplitude
        self._cont_phases = None   # (N,) radians

        # debug: count how many buffers have been filled
        self._debug_buffer_count = 0
        self._debug_oneshot_count = 0

        # audio capture (for recording)
        self._cap_lock   = threading.Lock()
        self._cap_active = False
        self._cap_chunks = []   # list of float32 arrays

    # capture

    def start_capture(self):
        """Begin accumulating every output buffer for later export."""
        with self._cap_lock:
            self._cap_active = True
            self._cap_chunks = []

    def stop_capture(self):
        """Stop accumulating output buffers (data is retained until get)."""
        with self._cap_lock:
            self._cap_active = False

    def get_captured_audio(self):
        """Return all captured samples as a single float32 array."""
        with self._cap_lock:
            if not self._cap_chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._cap_chunks)

    # one-shot

    def queue_oneshot(self, samples):
        """queue a pre-rendered sample buffer for playback."""
        arr = np.asarray(samples, dtype=np.float64)
        peak = np.max(np.abs(arr))
        with self.lock:
            self._oneshots.append([arr, 0])
            self._debug_oneshot_count += 1
            count = self._debug_oneshot_count
        print(f"[AudioState] Queued oneshot #{count}: "
              f"{len(arr)} samples, peak={peak:.4f}")

    # continuous-tone

    def set_continuous(self, freqs, amps):
        """Replace continuous tones. Phases preserved if count unchanged."""
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        amps_arr  = np.asarray(amps,  dtype=np.float64)
        with self.lock:
            if (self._cont_phases is None or
                    len(self._cont_phases) != len(freqs_arr)):
                self._cont_phases = np.zeros(len(freqs_arr), dtype=np.float64)
            self._cont_freqs = freqs_arr
            self._cont_amps  = amps_arr

    def clear_continuous(self):
        """Silence all continuous tones."""
        with self.lock:
            self._cont_freqs  = None
            self._cont_amps   = None
            self._cont_phases = None

    def clear_all(self):
        """Silence everything."""
        with self.lock:
            self._oneshots    = []
            self._cont_freqs  = None
            self._cont_amps   = None
            self._cont_phases = None

    # audio callback helper

    def fill_buffer(self, frames):
        """
        Generate `frames` samples of mixed audio. Returns float32 mono.
        Called on the AUDIO THREAD.
        """
        out = np.zeros(frames, dtype=np.float64)

        # snapshot under lock
        with self.lock:
            active_snapshot = self._oneshots
            self._oneshots  = []
            self._debug_buffer_count += 1
            buf_num = self._debug_buffer_count

            if self._cont_freqs is not None:
                c_freqs  = self._cont_freqs.copy()
                c_amps   = self._cont_amps.copy()
                c_phases = self._cont_phases.copy()
                has_continuous = True
            else:
                has_continuous = False

        # debug: confirm callback is running (every ~2 seconds)
        if buf_num % 172 == 1:  # 44100/512 ≈ 86 buffers/sec → every 2 sec
            n_shots = len(active_snapshot)
            print(f"[fill_buffer] buf#{buf_num}, oneshots={n_shots}, "
                  f"continuous={'yes' if has_continuous else 'no'}")

        # mix one-shot tones
        still_active = []
        for tone_samples, head in active_snapshot:
            remaining = len(tone_samples) - head
            if remaining <= 0:
                continue
            n = min(frames, remaining)
            out[:n] += tone_samples[head : head + n]
            new_head = head + n
            if new_head < len(tone_samples):
                still_active.append([tone_samples, new_head])

        # debug
        if len(active_snapshot) > 0:
            peak_out = float(np.max(np.abs(out)))
            print(f"[fill_buffer] Mixed {len(active_snapshot)} oneshots, "
                  f"peak_out={peak_out:.4f}")

        with self.lock:
            self._oneshots = still_active + self._oneshots

        # synthesise continuous tones
        if has_continuous and len(c_freqs) > 0:
            delta = 2.0 * np.pi * c_freqs / self.sample_rate
            idx   = np.arange(frames, dtype=np.float64)
            phase_matrix = (c_phases[:, np.newaxis]
                            + delta[:, np.newaxis] * idx[np.newaxis, :])
            sines = np.sin(phase_matrix) * c_amps[:, np.newaxis]
            out  += sines.sum(axis=0)

            new_phases = (c_phases + delta * frames) % (2.0 * np.pi)
            with self.lock:
                if self._cont_phases is not None:
                    self._cont_phases = new_phases

        # soft-clip
        out = np.tanh(out * 0.5)
        out_f32 = out.astype(np.float32)

        # capture output for recording
        with self._cap_lock:
            if self._cap_active:
                self._cap_chunks.append(out_f32.copy())

        return out_f32


class AudioEngine:
    """
    manages the real-time audio output stream.
    one instance shared across the whole application.
    """

    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.state       = AudioState(sample_rate)
        self._stream     = None

    def start(self):
        """open and start the audio output stream. Plays a test tone."""
        if self._stream is not None:
            return

        def _audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"[AudioEngine] status: {status}")
            try:
                buf = self.state.fill_buffer(frames)
                outdata[:, 0] = buf
                if outdata.shape[1] == 2:
                    outdata[:, 1] = buf
            except Exception as e:
                print(f"[AudioEngine] callback error: {e}")
                import traceback
                traceback.print_exc()
                outdata.fill(0)

        try:
            dev = sd.query_devices(kind='output')
            print(f"[AudioEngine] Output device: {dev['name']}")
            print(f"[AudioEngine] Device SR: {dev['default_samplerate']}")
            print(f"[AudioEngine] Opening: {self.sample_rate}Hz, "
                  f"buffer={self.buffer_size}, 2ch float32")

            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                channels=2,
                dtype='float32',
                callback=_audio_callback,
            )
            self._stream.start()

        except Exception as e:
            print(f"[AudioEngine] FAILED to start: {e}")
            import traceback
            traceback.print_exc()
            self._stream = None

    def stop(self):
        """Stop and close the audio stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[AudioEngine] stop error: {e}")
            self._stream = None
            print("[AudioEngine] Stream stopped")