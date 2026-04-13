"""
Sound banks and sonification pathway classes.

File layout

  OneShotSoundbank   — pre-rendered one-shot audio events (ding, gong)
  HallReverb         — Schroeder–Moorer large-hall reverb; wraps fill_buffer
  FitnessPathway     — converging-beat fitness sonification (any optimiser)
  PositionPathway    — PSO gbest-position spectral mapping (any optimiser)
  MutationPathway    — OPO accepted-mutation chord sonification (OPO only)
"""

import time
import threading
import numpy as np
from Synthesizer import AudioEngine


#  OneShotSoundbank — pre-rendered audio events

class OneShotSoundbank:
    """
    Pre-rendered one-shot audio events shared by all audification classes.

    Each method returns a numpy float64 array of audio samples ready to pass
    to AudioEngine.state.queue_oneshot().

    Sounds

    optimum_ding()        — bright triangle-wave strike; signals optimum found
    optimum_moved_gong()  — deep inharmonic metallic strike; signals target shift
    """

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def optimum_ding(self, base_freq=2200.0, duration_s=0.5, amplitude=0.65):
        """Bright triangle-like strike with odd harmonics and 1/k² rolloff."""
        n = int(duration_s * self.sr)
        t = np.arange(n) / self.sr

        signal = np.zeros(n, dtype=np.float64)
        for k in range(1, 14, 2):
            freq = base_freq * k
            if freq > self.sr / 2:
                break
            sign = (-1) ** ((k - 1) // 2)
            signal += sign / (k * k) * np.sin(2.0 * np.pi * freq * t)

        signal *= np.exp(-8.0 * t)

        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * amplitude

        return signal

    def optimum_moved_gong(self, base_freq=80.0, duration_s=1.5, amplitude=0.87):
        """Deep resonant metallic strike with inharmonic partials."""
        n = int(duration_s * self.sr)
        t = np.arange(n) / self.sr

        partials = [
            (1.00,  1.8,  1.00),
            (1.52,  2.5,  0.65),
            (2.28,  3.5,  0.45),
            (3.15,  5.0,  0.30),
            (4.72,  7.0,  0.18),
            (6.33, 10.0,  0.10),
        ]

        signal = np.zeros(n, dtype=np.float64)
        for ratio, decay, partial_amp in partials:
            freq = base_freq * ratio
            signal += partial_amp * np.exp(-decay * t) * np.sin(2.0 * np.pi * freq * t)

        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * amplitude

        attack = int(0.01 * self.sr)
        if 0 < attack < n:
            signal[:attack] *= np.linspace(0.0, 1.0, attack)

        return signal


#  HallReverb — Schroeder–Moorer hall reverb insert

class HallReverb:
    """
    Large-hall reverb using the classic Schroeder–Moorer topology:
    8 parallel feedback comb filters (with first-order low-pass damping in the
    feedback path) followed by 4 series allpass diffusers.

    :param sample_rate: must match AudioEngine.sample_rate (default 44100)
    :param rt60: reverberation time in seconds (default 3.0)
    :param wet: wet signal level 0–1 (default 0.38)
    :param damping: feedback low-pass coefficient 0–1; 0 = bright tail,
                    1 = maximum absorption (default 0.30)
    """

    _COMB_DELAYS_44K = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    _AP_DELAYS_44K   = [556, 441, 341, 225]

    def __init__(self, sample_rate=44100, rt60=3.0, wet=0.38, damping=0.30):
        self._wet     = float(wet)
        self._dry     = 1.0 - float(wet)
        self._damping = float(damping)
        self._damp1   = 1.0 - float(damping)

        scale   = sample_rate / 44100.0
        comb_d  = [max(2, int(d * scale)) for d in self._COMB_DELAYS_44K]
        ap_d    = [max(2, int(d * scale)) for d in self._AP_DELAYS_44K]

        # Comb feedback gains from RT60: g = 10^(−3 · delay_s / rt60)
        self._comb_g    = [
            10.0 ** (-3.0 * (d / sample_rate) / rt60)
            for d in comb_d
        ]
        self._n_combs   = len(comb_d)
        self._comb_bufs = [np.zeros(d, dtype=np.float64) for d in comb_d]
        self._comb_ptrs = [0] * self._n_combs
        self._comb_lp   = [0.0] * self._n_combs   # per-comb LP state

        self._ap_gain   = 0.5
        self._n_aps     = len(ap_d)
        self._ap_bufs   = [np.zeros(d, dtype=np.float64) for d in ap_d]
        self._ap_ptrs   = [0] * self._n_aps

    # core DSP — called on audio thread inside fill_buffer

    def _process(self, dry_f32):

        x     = dry_f32.astype(np.float64)
        n     = len(x)
        rev   = np.zeros(n, dtype=np.float64)

        damp  = self._damping
        damp1 = self._damp1

        # 8 parallel feedback comb filters
        for ci in range(self._n_combs):
            buf = self._comb_bufs[ci]
            d   = len(buf)
            g   = self._comb_g[ci]
            ptr = self._comb_ptrs[ci]
            lp  = self._comb_lp[ci]

            for i in range(n):
                delayed  = buf[ptr]
                lp       = delayed * damp1 + lp * damp   # LP in feedback path
                buf[ptr] = x[i] + lp * g
                rev[i]  += delayed
                ptr     += 1
                if ptr >= d:
                    ptr = 0

            self._comb_ptrs[ci] = ptr
            self._comb_lp[ci]   = lp

        rev *= (1.0 / self._n_combs)   # normalise parallel sum

        # 4 series allpass diffusers
        g_ap = self._ap_gain
        for ai in range(self._n_aps):
            buf = self._ap_bufs[ai]
            d   = len(buf)
            ptr = self._ap_ptrs[ai]

            for i in range(n):
                delayed  = buf[ptr]
                w        = rev[i] + g_ap * delayed
                buf[ptr] = w
                rev[i]   = delayed - g_ap * w
                ptr     += 1
                if ptr >= d:
                    ptr = 0

            self._ap_ptrs[ai] = ptr

        return (self._dry * dry_f32 + self._wet * rev).astype(np.float32)

    # wrap / unwrap

    def wrap(self, audio_state):
        """
        Monkey-patch audio_state.fill_buffer so every buffer passes through
        the hall reverb.  Safe to call repeatedly — guarded against re-wrapping.
        """
        if hasattr(audio_state, '_reverb_orig_fill'):
            return                         # already wrapped — no-op
        _orig    = audio_state.fill_buffer
        _process = self._process

        def _wrapped_fill(frames):
            return _process(_orig(frames))

        audio_state._reverb_orig_fill = _orig
        audio_state.fill_buffer       = _wrapped_fill

    def unwrap(self, audio_state):
        """Restore the original fill_buffer, removing the reverb insert."""
        orig = getattr(audio_state, '_reverb_orig_fill', None)
        if orig is not None:
            audio_state.fill_buffer = orig
            del audio_state._reverb_orig_fill


#  FitnessPathway — converging-beat fitness sonification

class FitnessPathway:
    """
    Sonifies the fitness trajectory of any optimiser as a pair of converging
    sine tones whose beating pattern reveals proximity to the optimum.

    Two continuous sine tones are maintained via AudioEngine.state.set_continuous():

        • Lower tone  — starts at FREQ_MIN (220 Hz) and rises  toward TARGET (440 Hz)
        • Upper tone  — starts at FREQ_MAX (880 Hz) and falls toward TARGET (440 Hz)

    Both tones are mapped linearly to the normalised fitness value f ∈ [0, 1]:

        f = 1  →  worst fitness  →  tones are maximally spread (220 Hz / 880 Hz)
        f = 0  →  best  fitness  →  tones converge on 440 Hz (A4), beating disappears
    """

    def __init__(self, audio_engine, amplitude=0.45):

        self.amplitude = amplitude
        self.engine = audio_engine

        self.freq_min = 392.0
        self.freq_max = 493.88
        self.freq_target = 440.0

        self._audio_state = self.engine.state

    def update(self, fitness, max_fitness):
        if max_fitness <= 0:
            return
        f = float(np.clip(fitness / max_fitness, 0.0, 1.0))
        self._set_tones(f)

    def stop(self):
        """Silence the pathway tones without stopping the AudioEngine."""
        self._audio_state.clear_continuous()

    def _set_tones(self, normalised_fitness):
        f          = normalised_fitness
        freq_lower = self.freq_target + f * (self.freq_min - self.freq_target)
        freq_upper = self.freq_target + f * (self.freq_max - self.freq_target)
        freqs = np.array([freq_lower, freq_upper], dtype=np.float64)
        amps  = np.array([self.amplitude, self.amplitude], dtype=np.float64)
        self._audio_state.set_continuous(freqs, amps)

#  PositionPathway — PSO gbest-position spectral mapping

class PositionPathway:
    """
    Maps a global-best candidate position to n continuous sine tones — one per
    dimension — and pushes them to the AudioEngine each iteration.

    Frequency bracketing

    Bracket edges follow the formula:
        e[k] = 100 × 2^(k·(1−t)) × (k+1)^t     k = 0 … n

    t = 0  →  pure octave spacing: e = [100, 200, 400, 800, …]
    t = 1  →  harmonic series:     e = [100, 200, 300, 400, …]

    t is computed per n by solving e[n] = 20 000 Hz analytically:
        t = (ln 200 − n·ln 2) / (ln(n+1) − n·ln 2)

    This gives t = 0 for small n (pure octave) and rises toward t ≈ 1 for
    large n (harmonic series).  The lowest bracket starts at 100 Hz for
    clarity — the previous 50 Hz base was too muddy.

    Diversity effects

    update() accepts an optional `diversity` float in [0, 1] representing
    normalised swarm dispersion.  When > 0 it blends in:
        • Chorus  — two detuned copies (±spread) of every main tone, amplitude
                    proportional to diversity.
        • Noise   — 20 fixed-random sine tones across the band, amplitude
                    scaled by diversity × _NOISE_AMP_MAX (kept subtle).

    Amplitude decay

    amp[d] = exp(−α · d),  normalised so all amps sum to _AMP_TARGET = 0.6.
    α = α_min + (α_max − α_min) × √t
    Low t (small n):  α ≈ 0.05 → nearly flat   → chord-like impression
    High t (large n): α = 2.0  → strong taper  → timbral impression
    """

    def __init__(self, n, r, audio_engine):
        self.n            = n
        self.r            = r
        self._audio_state = audio_engine.state

        self._amp_alpha_min = 0.05
        self._amp_alpha_max = 0.8  # reduced: gentler taper so upper partials stay audible
        self._amp_target = 0.6

        # Diversity effects (noise + chorus)
        self._n_noise_tones = 20  # random sine tones that approximate band noise
        self._noise_amp_max = 0.04  # peak noise amplitude at full diversity
        self._chorus_spread_max = 0.030  # max detune ratio (±3 %) for chorus copies
        self._chorus_amp_scale = 0.25  # chorus-copy amplitude as fraction of main tone

        self._recompute_brackets()

    # public

    def update(self, gbest_position, diversity=0.0):
        """
        Recompute tones from the current global-best position and push to audio.

        gbest_position : array-like (n,)  — current global-best candidate
        diversity      : float in [0, 1]  — normalised swarm dispersion.
                         0 = fully converged (clean), 1 = maximally dispersed.
                         When > 0, subtle chorus and noise are blended in.
        """
        freqs, amps = self._compute_tones(gbest_position)
        if len(freqs) == 0:
            return
        if diversity > 0.0:
            freqs, amps = self._apply_diversity_effects(freqs, amps, diversity)
        self._audio_state.set_continuous(freqs, amps)

    def stop(self):
        self._audio_state.clear_continuous()

    # internal

    def _recompute_brackets(self):
        """Compute bracket edges and per-dimension amplitudes from n and r."""
        n = max(self.n, 1)

        # Squish factor t: 0 = pure octave, 1 = harmonic series
        # Solve e[n] = 20000 Hz with base 100: 100·2^(n(1-t))·(n+1)^t = 20000
        #   → t = (ln 200 − n·ln 2) / (ln(n+1) − n·ln 2)
        if n <= 8:
            t = 0.0
        else:
            ln2 = np.log(2.0)
            num = np.log(200.0) - n * ln2
            den = np.log(float(n + 1)) - n * ln2
            t   = float(np.clip(num / den, 0.0, 1.0))

        self._squish_t = t

        # Bracket edges: e[k] = 100 · 2^(k(1-t)) · (k+1)^t
        k     = np.arange(n + 1, dtype=np.float64)
        edges = 100.0 * (2.0 ** (k * (1.0 - t))) * ((k + 1.0) ** t)
        edges = np.minimum(edges, 20000.0)

        self._bracket_low  = edges[:-1]
        self._bracket_high = edges[1:]

        # Amplitude decay: α grows with √t
        alpha    = self._amp_alpha_min + (self._amp_alpha_max - self._amp_alpha_min) * t
        d        = np.arange(n, dtype=np.float64)
        raw_amps = np.exp(-alpha * d)
        total    = raw_amps.sum()
        self._dim_amps = raw_amps / total * self._amp_target if total > 0 else raw_amps

        # Fixed random noise frequencies (consistent between calls, diversity scales amp)
        rng = np.random.default_rng(seed=42)
        self._noise_freqs = rng.uniform(100.0, 4000.0, self._n_noise_tones)

    def _compute_tones(self, gbest_position):
        """Map gbest_position to (freqs, amps) arrays of length n."""
        pos   = np.asarray(gbest_position, dtype=np.float64)
        n     = min(len(pos), self.n)
        r_max = max(float(self.r - 1), 1.0)

        pos_frac = np.clip(pos[:n] / r_max, 0.0, 1.0)
        low      = self._bracket_low[:n]
        high     = self._bracket_high[:n]
        ratio    = np.where(high > low, high / low, 1.0)
        freqs    = low * (ratio ** pos_frac)
        amps     = self._dim_amps[:n].copy()

        return freqs, amps

    def _apply_diversity_effects(self, freqs, amps, diversity):
        """
        Blend in chorus copies and band noise scaled by swarm diversity [0, 1].

        Chorus:  two detuned copies of each main tone (one sharp, one flat)
                 at ±diversity × _CHORUS_SPREAD_MAX.  Amplitude = diversity
                 × _CHORUS_AMP_SCALE × main-tone amplitude.
        Noise:   _N_NOISE_TONES fixed-random sine tones across 100–4000 Hz,
                 each at amplitude diversity × _NOISE_AMP_MAX / N_NOISE_TONES.
        """
        spread      = diversity * self._chorus_spread_max
        chorus_amp  = diversity * self._chorus_amp_scale
        noise_amp_each = (self._noise_amp_max * diversity
                          / max(self._n_noise_tones, 1))

        extra_f = []
        extra_a = []

        # Chorus: sharp copy (+spread) and flat copy (−spread)
        for f, a in zip(freqs, amps):
            extra_f.append(f * (1.0 + spread))
            extra_a.append(a * chorus_amp)
            extra_f.append(f * (1.0 - spread))
            extra_a.append(a * chorus_amp)

        # Noise tones
        extra_f.extend(self._noise_freqs.tolist())
        extra_a.extend([noise_amp_each] * self._n_noise_tones)

        combined_freqs = np.concatenate([freqs, np.array(extra_f, dtype=np.float64)])
        combined_amps  = np.concatenate([amps,  np.array(extra_a, dtype=np.float64)])
        return combined_freqs, combined_amps


#  MutationPathway

class MutationPathway:
    """
    Sounds a C-major triad (C4 / E4 / G4) whenever a mutation is accepted.

    Each note is built from 16 partials with exponentially decaying amplitudes.
    The partial multipliers start as random deviations from integer harmonics
    (±0.5) and converge toward perfect harmonics as the fitness improves.
    When fitness worsens (e.g. dynamic optimum shift) the inharmonicity grows.

    Envelope / fast-mutation handling
    The chord decays over ~1.5 s via a 120 Hz background thread.  If a new
    mutation arrives before the decay finishes, the amplitude snaps back to
    full and the partial multipliers glide quickly to the new values — no new
    envelope is spawned, preventing oneshot-queue overload.

    Reverb
    The Hamming distance of the mutation normalised by search-space size
    determines the wet level of a short pre-rendered reverb tail.
    """

    def __init__(self, n, r, audio_engine):
        self.n            = n
        self.r            = r
        self._engine      = audio_engine
        self._audio_state = audio_engine.state

        self.n_partials = 16
        self.chord_notes = np.array([261.63, 329.63, 392.00])  # C4  E4  G4
        self.env_duration = 1.5  # s  — full chord decay time
        self.attack_time = 0.022  # s  — attack ramp (slightly longer for perceptibility)
        self.update_hz = 120  # Hz — envelope-thread update rate
        self.glide_time = 0.04  # s  — partial-multiplier glide on fast retrigger
        self.chord_amplitude = 0.6  # per-note peak amplitude
        self.max_dev_scale = 3.0  # deviation scale ceiling

        # partial / spectral design
        self._harm_mults = np.arange(1, self.n_partials + 1, dtype=np.float64)
        _k   = np.arange(self.n_partials, dtype=np.float64)
        _raw = np.exp(-0.5 * _k)
        self._partial_amps = _raw / _raw.sum()   # shape (16,), sums to 1

        self._base_deviations = np.random.uniform(
            -0.5, 0.5, (3, self.n_partials)
        )
        # Lock the fundamental (partial index 0) to zero deviation so the
        # chord tones C4 / E4 / G4 stay at their exact pitches.  Only the
        # overtones (partials 2–16, indices 1–15) carry the inharmonicity.
        self._base_deviations[:, 0] = 0.0

        # envelope decay constant
        _decay_window = max(self.env_duration - self.attack_time, 1e-6)
        self._decay_k = 6.908 / _decay_window   # 6.908 ≈ ln(1000) = -ln(1e-3)

        # fitness / deviation state
        self._deviation_scale = 1.0
        self._search_space    = max(n * (r - 1), 1)

        # envelope state (shared with background thread)
        self._lock             = threading.Lock()
        self._env_active       = False
        self._env_elapsed      = 0.0
        self._env_amp          = 0.0
        self._retrigger_floor  = 0.0

        _init = self._harm_mults[np.newaxis, :] + np.zeros((3, 1))
        self._playing_mults = _init.copy()
        self._target_mults  = _init.copy()

        # start envelope thread
        self._stop_thread = False
        self._env_thread  = threading.Thread(
            target=self._envelope_loop, daemon=True
        )
        self._env_thread.start()

    # public

    def on_mutation(self, current_candidate, current_fitness,
                    old_candidate, old_fitness):
        """
        Call this whenever a mutation is accepted (mutant_taken == True).

        Updates the inharmonicity scale, triggers/retriggeres the chord, and
        queues a reverb tail proportional to the mutation's search-space distance.
        """

        self._deviation_scale = float(np.clip(
            current_fitness / self._search_space * self.max_dev_scale,
            0.0, self.max_dev_scale,
        ))

        # target partial multipliers
        scaled_dev = self._base_deviations * self._deviation_scale
        new_target = self._harm_mults[np.newaxis, :] + scaled_dev

        # reverb amount from normalised Hamming distance
        dist = float(np.sum(np.abs(
            np.asarray(current_candidate, dtype=np.float64)
            - np.asarray(old_candidate,   dtype=np.float64)
        )))
        reverb_amount = float(np.clip(dist / self._search_space, 0.0, 1.0))

        # trigger / retrigger chord
        with self._lock:
            if self._env_active:
                self._retrigger_floor = float(np.clip(self._env_amp, 0.0, 1.0))
                self._env_elapsed     = 0.0
                self._target_mults    = new_target
            else:
                # Fresh start from silence
                self._env_active      = True
                self._env_elapsed     = 0.0
                self._env_amp         = 0.0
                self._retrigger_floor = 0.0
                self._playing_mults   = new_target.copy()
                self._target_mults    = new_target.copy()

        # reverb tail
        if reverb_amount > 0.02:
            tail = self._render_reverb_tail(new_target, reverb_amount)
            self._audio_state.queue_oneshot(tail)

    def stop(self):
        """Stop the envelope thread and silence the chord."""
        self._stop_thread = True
        self._audio_state.clear_continuous()

    # envelope background thread

    def _envelope_loop(self):
        """
        Runs at UPDATE_HZ.  Drives the chord envelope and glides partial
        multipliers toward their target values on fast retriggers.
        Single lock section prevents retrigger writes from being clobbered.
        """
        dt      = 1.0 / self.update_hz
        decay_k = self._decay_k   # pre-computed in __init__; amplitude → 1e-3 at ENV_DURATION
        sr      = self._audio_state.sample_rate
        nyquist = sr / 2.0

        while not self._stop_thread:
            do_clear = False
            freqs    = None
            amps     = None

            with self._lock:
                if self._env_active:
                    elapsed = self._env_elapsed
                    playing = self._playing_mults
                    target  = self._target_mults

                    floor = self._retrigger_floor
                    if elapsed < self.attack_time:
                        new_amp = floor + (1.0 - floor) * (elapsed / self.attack_time)
                    else:
                        new_amp = np.exp(-decay_k * (elapsed - self.attack_time))

                    new_elapsed = elapsed + dt

                    # glide multipliers toward target
                    alpha     = min(dt / self.glide_time, 1.0)
                    new_mults = playing + alpha * (target - playing)

                    # 3×16 = 48 continuous sines
                    freqs = (self.chord_notes[:, np.newaxis] * new_mults).ravel()
                    amps  = np.tile(
                        self._partial_amps * (new_amp * self.chord_amplitude), 3
                    )
                    amps[freqs >= nyquist] = 0.0

                    done = (elapsed >= self.attack_time and new_amp < 1e-3)

                    self._env_elapsed   = new_elapsed
                    self._env_amp       = new_amp
                    self._playing_mults = new_mults
                    if done:
                        self._env_active      = False
                        self._env_amp         = 0.0
                        self._retrigger_floor = 0.0
                        do_clear              = True

            if do_clear:
                self._audio_state.clear_continuous()
            elif freqs is not None:
                self._audio_state.set_continuous(freqs, amps)

            time.sleep(dt)

    # reverb tail

    def _render_reverb_tail(self, mults, reverb_amount):
        """
        Pre-render an exponentially decaying reverb tail for the current chord.
        Uses the first 6 partials per note for computational speed.
        """
        sr      = self._audio_state.sample_rate
        nyquist = sr / 2.0
        rt      = 0.3 + reverb_amount * 1.4    # reverb time 0.3–1.7 s
        n       = int(rt * sr)
        t       = np.arange(n, dtype=np.float64) / sr

        pre_n   = int(0.025 * sr)               # 25 ms pre-delay
        decay_r = 5.0 / rt
        env     = np.zeros(n, dtype=np.float64)
        if pre_n < n:
            env[pre_n:] = np.exp(-decay_r * (t[pre_n:] - t[pre_n]))
            onset_n = min(int(0.006 * sr), n - pre_n)   # 6 ms ramp
            if onset_n > 1:
                env[pre_n : pre_n + onset_n] *= np.linspace(0.0, 1.0, onset_n)

        signal = np.zeros(n, dtype=np.float64)
        for i, note_freq in enumerate(self.chord_notes):
            for j in range(6):
                freq = float(note_freq * mults[i, j])
                if freq >= nyquist:
                    continue
                signal += self._partial_amps[j] / 3.0 * np.sin(2.0 * np.pi * freq * t)

        signal *= env
        peak = np.max(np.abs(signal))
        if peak > 1e-6:
            signal = signal / peak * (reverb_amount * 0.2)

        return signal

