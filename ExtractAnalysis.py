"""
Recording and analysis.

gets per-iteration optimisation data and synthesized audio, then exports:
    • An MP3 of the synthesized audio.
    • A PNG analysis report with three vertically-stacked panels and
      a parameter summary at the top.

layout:
  1.  Fitness / Global-Best vs Iteration
  2.  Audio spectrogram  (1024 log-spaced bins, dB scale)
      Aligned to iteration 0 — leading audio before play is pressed is
      trimmed away automatically.
  3.  OPO: successful mutation tick-marks
      PSO: swarm diversity (mean std-dev of positions)
"""

import numpy as np
import datetime
import os
import time

try:
    import soundfile as _sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False
    print("[ExtractAnalysis] soundfile not available — audio won't be saved to MP3")

try:
    import scipy.signal as _sig
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[ExtractAnalysis] scipy not available — spectrogram will be blank")

try:
    import matplotlib as _mpl
    _mpl.use('Agg')                      # non-interactive, no display needed
    import matplotlib.pyplot as _plt
    import matplotlib.gridspec as _gs
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[ExtractAnalysis] matplotlib not available — PNG export disabled")

# spectrogram colour map: black → dark-blue → app-blue → app-yellow → white
if _HAS_MPL:
    _SPEC_CMAP = _LSC.from_list('dso_spec', [
        (0.00, (0.00, 0.00, 0.00)),          # black  (silence)
        (0.25, (0.04, 0.09, 0.32)),          # very dark blue
        (0.55, (130/255, 200/255, 1.00)),    # app blue  (130, 200, 255)
        (0.80, (1.00, 200/255, 0.00)),       # app yellow (255, 200,   0)
        (1.00, (1.00, 1.00, 1.00)),          # white  (peak)
    ])
else:
    _SPEC_CMAP = None

# matplotlib rcParams for dark theme
_DARK_RC = {
    'font.family':      'monospace',
    'font.monospace':   ['Courier New', 'Courier', 'monospace'],
    'font.size':        10,
    'text.color':       'white',
    'axes.facecolor':   'black',
    'axes.edgecolor':   '#555555',
    'axes.labelcolor':  'white',
    'xtick.color':      'white',
    'ytick.color':      'white',
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'grid.color':       '#2a2a2a',
    'grid.linewidth':   0.5,
    'figure.facecolor': 'black',
    'figure.edgecolor': 'black',
    'lines.linewidth':  1.5,
    'axes.titlecolor':  'white',
    'axes.titlesize':   11,
    'axes.titlepad':    6,
    'axes.labelsize':   10,
}

# App colours (matching live visualisations)
_COL_YELLOW = (1.0,  200/255, 0.0)      # (255, 200,   0) gbest / mutations
_COL_BLUE   = (130/255, 200/255, 1.0)   # (130, 200, 255) diversity / app-blue
_COL_WHITE  = (1.0,  1.0,  1.0)
_COL_RED_V  = '#FF4444'                  # optimum-moved vlines


#  ExtractAnalysis — main recorder / exporter class

class ExtractAnalysis:
    """
    Records optimisation iteration data and synthesized audio, then exports
    an MP3 and a PNG analysis report.

    Typical usage
        rec = ExtractAnalysis('OPO')           # or 'PSO'
        rec.set_params({'r': 2, 'n': 10, …})
        rec.set_audio_engine(engine)
        rec.start_recording()
        # … per iteration …
        rec.record_opo_step(…)
        # … when done …
        rec.stop_and_export(mp3_path, png_path)
    """

    SAMPLE_RATE = 44100

    def __init__(self, algorithm_type):
        self.algorithm_type   = algorithm_type   # 'OPO' or 'PSO'
        self.steps            = []
        self._params          = {}
        self._audio_engine    = None

        self.show_optimum_lines = True

        # Timing for spectrogram alignment
        self._rec_start_time  = None   # wall-clock when start_recording() called
        self._play_start_time = None   # wall-clock when first opt step recorded

    # configuration

    def set_params(self, params_dict):
        """Store algorithm / problem parameters for the PNG header."""
        self._params = dict(params_dict)

    def set_audio_engine(self, engine):
        """Pass the AudioEngine so we can capture what it synthesizes."""
        self._audio_engine = engine

    #  recording

    def record_opo_step(self, iteration, current_fitness, old_fitness,
                        mutant_taken, optimum_moved, synth_freq=None,
                        ding_fired=False, gong_fired=False, first_ding=None):
        if self._play_start_time is None:
            self._play_start_time = time.time()
        self.steps.append({
            'iteration':       iteration,
            'current_fitness': current_fitness,
            'old_fitness':     old_fitness,
            'mutant_taken':    mutant_taken,
            'optimum_moved':   optimum_moved,
            'synth_freq':      synth_freq,
            'ding_fired':      ding_fired,
            'gong_fired':      gong_fired,
            'first_ding':      first_ding,
        })

    def record_pso_step(self, iteration, gbest_value, position_fits,
                        positions, optimum_moved, gong_fired=False,
                        tone_freqs=None, tone_amps=None):
        if self._play_start_time is None:
            self._play_start_time = time.time()
        self.steps.append({
            'iteration':     iteration,
            'gbest_value':   gbest_value,
            'position_fits': position_fits.copy(),
            'positions':     positions.copy(),
            'optimum_moved': optimum_moved,
            'gong_fired':    gong_fired,
            'tone_freqs':    tone_freqs.copy() if tone_freqs is not None else None,
            'tone_amps':     tone_amps.copy()  if tone_amps  is not None else None,
        })

    # recording start

    def start_recording(self):
        """
        Start audio capture.  Call this when the record button is pressed.
        """
        self._rec_start_time = time.time()
        if self._audio_engine is not None:
            self._audio_engine.state.start_capture()
            print("[ExtractAnalysis] Audio capture started")
        else:
            print("[ExtractAnalysis] WARNING: no audio engine — audio won't be captured")

    # stop + full export

    def stop_and_export(self, mp3_path, png_path):

        # 1. Stop audio capture and retrieve samples
        audio = np.zeros(0, dtype=np.float32)
        if self._audio_engine is not None:
            self._audio_engine.state.stop_capture()
            audio = self._audio_engine.state.get_captured_audio()
            duration_s = len(audio) / self.SAMPLE_RATE
            print(f"[ExtractAnalysis] Audio captured: {len(audio)} samples "
                  f"({duration_s:.1f}s)")
        else:
            print("[ExtractAnalysis] WARNING: no audio engine — MP3 will be silent")

        # 2. Save MP3
        self._save_mp3(audio, mp3_path)

        # 3. Export PNG (spectrogram trimmed to play-start)
        self.export_png(png_path, audio=audio)

    # MP3 export

    def _save_mp3(self, audio, mp3_path):
        """
        Write the captured audio to an MP3 file using soundfile (via libsndfile).
        Falls back to WAV if MP3 encoding is not supported.
        """
        if len(audio) == 0:
            print("[ExtractAnalysis] Audio array is empty — MP3 not saved")
            return
        if not _HAS_SF:
            print("[ExtractAnalysis] soundfile not installed — MP3 not saved")
            return

        try:
            # soundfile supports MP3 write when built with libsndfile ≥ 1.1.0;
            # fall back to WAV if not available.
            _sf.write(mp3_path, audio, self.SAMPLE_RATE)
            print(f"[ExtractAnalysis] MP3 saved → {mp3_path} "
                  f"({os.path.getsize(mp3_path)} bytes)")
        except Exception as exc:
            # Fallback: save as WAV alongside the intended MP3 path
            wav_path = os.path.splitext(mp3_path)[0] + '.wav'
            print(f"[ExtractAnalysis] MP3 write failed ({exc}); "
                  f"saving WAV instead → {wav_path}")
            try:
                _sf.write(wav_path, audio, self.SAMPLE_RATE, subtype='PCM_16')
                print(f"[ExtractAnalysis] WAV saved → {wav_path}")
            except Exception as exc2:
                print(f"[ExtractAnalysis] WAV fallback also failed: {exc2}")

    #  PNG export

    def export_png(self, filepath, audio=None):
        """
        `audio` is an optional float32 array of the synthesized audio.
        If omitted, audio is retrieved from the attached engine (if any).
        """
        if not _HAS_MPL:
            print("[ExtractAnalysis] matplotlib unavailable — PNG not exported")
            return

        if audio is None:
            audio = np.zeros(0, dtype=np.float32)
            if self._audio_engine is not None:
                audio = self._audio_engine.state.get_captured_audio()

        if len(self.steps) == 0:
            self._write_empty_png(filepath)
            return

        with _mpl.rc_context(_DARK_RC):
            if self.algorithm_type == 'OPO':
                self._build_opo_fig(filepath, audio)
            else:
                self._build_pso_fig(filepath, audio)

    # empty / error fallback

    def _write_empty_png(self, filepath):
        with _mpl.rc_context(_DARK_RC):
            fig, ax = _plt.subplots(1, 1, figsize=(12, 2))
            ax.text(0.5, 0.5, 'No data recorded.',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='#888888')
            ax.axis('off')
            fig.savefig(filepath, format='png', dpi=150,
                        bbox_inches='tight', facecolor='black')
            _plt.close(fig)

    # spectrogram audio alignment

    def _trim_audio_to_play_start(self, audio):
        """
        Remove audio samples recorded before the play button was pressed,
        so the spectrogram x-axis (time in seconds) aligns with iteration 0.
        """
        if (self._rec_start_time is not None and
                self._play_start_time is not None and
                self._play_start_time > self._rec_start_time):
            offset_s       = self._play_start_time - self._rec_start_time
            offset_samples = int(offset_s * self.SAMPLE_RATE)
            if 0 < offset_samples < len(audio):
                print(f"[ExtractAnalysis] Trimming {offset_s:.2f}s "
                      f"({offset_samples} samples) of pre-play audio "
                      f"from spectrogram")
                return audio[offset_samples:]
        return audio

    # figure skeleton

    def _make_figure(self):
        """
        Create a 14×11 inch figure with three subplots, reserving the top
        18 % for the parameters header.  Returns (fig, ax_fit, ax_spec, ax_algo).
        """
        fig = _plt.figure(figsize=(14, 11), facecolor='black')
        grid = _gs.GridSpec(
            3, 1,
            figure=fig,
            height_ratios=[1.1, 1.5, 1.1],
            hspace=0.55,
            left=0.07, right=0.97, top=0.82, bottom=0.06,
        )
        ax0 = fig.add_subplot(grid[0])
        ax1 = fig.add_subplot(grid[1])
        ax2 = fig.add_subplot(grid[2])
        return fig, ax0, ax1, ax2

    # parameters header

    def _add_header(self, fig, ts):
        """Render the run-info + parameter block above the panels."""
        alg_label = ('(1+1) Evolutionary Algorithm'
                     if self.algorithm_type == 'OPO'
                     else 'Particle Swarm Optimisation')

        fig.text(0.5, 0.975,
                 f'DynSonOpt  ·  {alg_label}  ·  {ts}',
                 ha='center', va='top',
                 fontsize=12, color='white',
                 fontfamily='monospace', fontweight='bold')

        fig.text(0.5, 0.951,
                 f'{len(self.steps)} iterations recorded',
                 ha='center', va='top',
                 fontsize=9, color='#999999',
                 fontfamily='monospace')

        if self._params:
            pairs    = [f'{k}: {v}' for k, v in self._params.items()]
            row_size = 4
            rows     = [pairs[i:i + row_size]
                        for i in range(0, len(pairs), row_size)]
            y        = 0.927
            for row in rows:
                fig.text(0.5, y, '    '.join(row),
                         ha='center', va='top',
                         fontsize=9, color='#aaaaaa',
                         fontfamily='monospace')
                y -= 0.028

    # axis styling

    @staticmethod
    def _style_axis(ax, title, xlabel='Iteration', ylabel=''):
        ax.set_facecolor('black')
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.tick_params(colors='white', which='both')
        for sp in ax.spines.values():
            sp.set_edgecolor('#555555')
        ax.grid(True, color='#2a2a2a', linewidth=0.5,
                linestyle='--', alpha=0.8)

    # optimum-moved vlines

    def _add_optimum_vlines(self, ax):
        """Draw dashed red vertical lines at every optimum-move event.
        but not for DynamicOneMax."""
        if not self.show_optimum_lines:
            return
        for s in self.steps:
            if s.get('optimum_moved'):
                ax.axvline(s['iteration'],
                           color=_COL_RED_V, lw=0.7,
                           alpha=0.65, linestyle='--', zorder=2)

    # spectrogram panel

    def _plot_spectrogram(self, ax, audio):
        """
        Compute a log-frequency spectrogram and render it into *ax*.

        Expects audio already trimmed to play-start (via _trim_audio_to_play_start).
        1024 log-spaced frequency bins, dB power, custom yellow/white/blue
        colour map matching the live PSO visualisation palette.
        """
        if not _HAS_SCIPY or len(audio) < 512:
            msg = ('No audio captured\n(OSC mode, or recording too short)'
                   if len(audio) < 512 else
                   'scipy unavailable — spectrogram skipped')
            ax.text(0.5, 0.5, msg,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='#666666')
            self._style_axis(ax, 'Audio Spectrogram',
                             xlabel='Time (s)', ylabel='Frequency (Hz)')
            return

        fs      = self.SAMPLE_RATE
        nperseg = 2048
        hop     = 512

        # STFT → complex amplitudes, shape (nperseg/2+1, T)
        f_lin, t_stft, Zxx = _sig.stft(
            audio.astype(np.float64),
            fs=fs, nperseg=nperseg,
            noverlap=nperseg - hop,
            boundary='zeros',
        )

        # Power spectrum → dB
        power_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

        # Resample to 1024 log-spaced frequency bins (20 Hz … Nyquist)
        f_min  = 20.0
        f_max  = float(fs) / 2.0
        n_bins = 1024
        f_log  = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)

        log_spec = np.zeros((n_bins, len(t_stft)), dtype=np.float32)
        for ti in range(len(t_stft)):
            log_spec[:, ti] = np.interp(f_log, f_lin, power_db[:, ti])

        # Clip to top 50 dB of dynamic range
        db_max = float(log_spec.max())
        db_min = float(max(log_spec.min(), db_max - 50.0))
        log_spec = np.clip(log_spec, db_min, db_max)

        # Draw
        mesh = ax.pcolormesh(t_stft, np.arange(n_bins), log_spec,
                             cmap=_SPEC_CMAP, shading='auto',
                             vmin=db_min, vmax=db_max)
        mesh.set_rasterized(True)

        # Y-axis: Hz labels at musically meaningful log positions
        tick_hz  = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        tick_idx = [int(np.searchsorted(f_log, fh))
                    for fh in tick_hz if f_min <= fh <= f_max]
        tick_lbl = [f'{fh}' if fh < 1000 else f'{fh // 1000}k'
                    for fh in tick_hz if f_min <= fh <= f_max]
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(tick_lbl)

        self._style_axis(ax,
                         'Audio Spectrogram  (1024 log-spaced bins, dB)',
                         xlabel='Time (s)', ylabel='Frequency (Hz)')

    #  OPO

    def _build_opo_fig(self, filepath, audio):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig, ax0, ax1, ax2 = self._make_figure()
        self._add_header(fig, ts)

        iters   = [s['iteration']       for s in self.steps]
        fitness = [s['current_fitness'] for s in self.steps]

        # Panel 1: Fitness vs Iteration
        ax0.plot(iters, fitness, color=_COL_WHITE, lw=1.2, zorder=3)
        ax0.set_xlim(iters[0], iters[-1])
        self._add_optimum_vlines(ax0)
        self._style_axis(ax0, 'Fitness vs Iteration', ylabel='Fitness')

        # Panel 2: Spectrogram (trimmed to play-start)
        self._plot_spectrogram(ax1, self._trim_audio_to_play_start(audio))

        # Panel 3: Successful Mutations
        for s in self.steps:
            if s['mutant_taken']:
                ax2.axvline(s['iteration'],
                            color=_COL_YELLOW, lw=0.55, alpha=0.85, zorder=3)
        self._add_optimum_vlines(ax2)

        ax2.set_xlim(min(iters), max(iters))
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.spines['left'].set_visible(False)
        vline_note = '' if not self.show_optimum_lines else '  (yellow = accepted,  red dashed = optimum moved)'
        self._style_axis(ax2, 'Successful Mutations' + vline_note)

        self._save_fig(fig, filepath)

    #  PSO

    def _build_pso_fig(self, filepath, audio):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig, ax0, ax1, ax2 = self._make_figure()
        self._add_header(fig, ts)

        iters      = [s['iteration']   for s in self.steps]
        gbest_vals = [s['gbest_value'] for s in self.steps]

        # Panel 1: Global Best vs Iteration
        ax0.plot(iters, gbest_vals, color=_COL_WHITE, lw=1.2, zorder=3)
        ax0.set_xlim(iters[0], iters[-1])
        self._add_optimum_vlines(ax0)
        self._style_axis(ax0, 'Global Best Fitness vs Iteration',
                         ylabel='Global Best')

        # Panel 2: Spectrogram
        self._plot_spectrogram(ax1, self._trim_audio_to_play_start(audio))

        # Panel 3: Swarm Diversity
        diversities = [
            float(np.std(s['positions'], axis=0).mean())
            for s in self.steps
        ]
        ax2.plot(iters, diversities, color=_COL_BLUE, lw=1.2, zorder=3)
        ax2.set_xlim(iters[0], iters[-1])
        self._add_optimum_vlines(ax2)
        div_title = ('Swarm Diversity  (mean std-dev of particle positions)'
                     if not self.show_optimum_lines else
                     'Swarm Diversity  (mean std-dev of positions,  red dashed = optimum moved)')
        self._style_axis(ax2, div_title, ylabel='Diversity')

        self._save_fig(fig, filepath)

    # figure save

    @staticmethod
    def _save_fig(fig, filepath):
        fig.savefig(filepath, format='png', dpi=150,
                    bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        _plt.close(fig)
        print(f"[ExtractAnalysis] PNG saved → {filepath}")
