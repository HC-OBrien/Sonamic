"""
Main application:
    • Title (top-left), Version 1.1 (top-right)
    • Problem + algorithm + sonification pathway dropdowns (under title)
    • Central panel (visualisation renders here)
    • Play/Pause + Stop transport (bottom centre)
    • Sound engine toggle Py/OSC (bottom-left, GREYED OUT while running)
"""

from Audiovisualisations import AudioVisualiseOPO, AudioVisualisePSO
from DynamicOptimisers import OPO, PSO
from DynamicProblems import DynamicOneMax, RandomIntStr, AllToNone
from ExtractAnalysis import ExtractAnalysis
from Synthesizer import AudioEngine
import datetime
import os
import py5


class DynamicSonOptApp:

    def __init__(self):
        # shared audio engine
        self.audio_engine = AudioEngine()
        self.audio_engine.start()

        # window dimensions
        self.width  = 1400
        self.height = 800

        # simulation state
        self.is_running   = False
        self.is_paused    = False
        self.sound_engine = 'Py'     # 'Py' or 'OSC'

        # dropdown options
        self.problem_options = ["DynamicOneMax", "RandomIntStr", "AllToNone"]
        self.algorithm_options = [
            "[1+1] Evolutionary Algorithm",
            "Particle Swarm Optimisation",
        ]
        self.pathway_options = [
            "Fitness Pathway",
            "Position Pathway",
            "Mutation Pathway",
        ]
        # Internal keys matching the pathway display names
        self._pathway_keys = {
            "Fitness Pathway":  "fitness",
            "Position Pathway": "position",
            "Mutation Pathway": "mutation",
        }

        self.problem_selected_index   = None
        self.algorithm_selected_index = None
        self.pathway_selected_index   = None
        self.problem_dropdown_open    = False
        self.algorithm_dropdown_open  = False
        self.pathway_dropdown_open    = False

        # dropdown
        self.problem_dropdown_x = 20
        self.problem_dropdown_y = 105
        self.problem_dropdown_w = 230
        self.problem_dropdown_h = 30

        self.algorithm_dropdown_x = 260
        self.algorithm_dropdown_y = 105
        self.algorithm_dropdown_w = 310
        self.algorithm_dropdown_h = 30

        self.pathway_dropdown_x = 580
        self.pathway_dropdown_y = 105
        self.pathway_dropdown_w = 230
        self.pathway_dropdown_h = 30

        # central panel
        self.panel_x = 20
        self.panel_y = 150
        self.panel_w = self.width - 40
        self.panel_h = self.height - 270

        # sound engine toggle
        self.toggle_x = 20
        self.toggle_y = self.height - 85
        self.toggle_w = 120
        self.toggle_h = 26

        # record button
        self.record_btn_x = self.width - 60
        self.record_btn_y = self.height - 55
        self.record_btn_r = 16          # circle radius

        # recording state
        self.is_recording = False
        self.recorder     = None

        # active view
        self.view = None
        self.font = None

    def settings(self):
        py5.size(self.width, self.height, py5.P3D)

    def setup(self):
        # Set a fixed window title so screen recording can find this window
        py5.get_surface().set_title("DynSonOpt")
        self.font = py5.create_font("Courier New", 20)
        py5.text_font(self.font)
        py5.frame_rate(60)

    def draw(self):
        py5.background(0)

        # central panel border
        py5.hint(py5.DISABLE_DEPTH_TEST)
        self._draw_central_panel_border()

        # view content
        if self.view is not None:
            if self.is_running and not self.is_paused:
                self.view.draw_and_sound()
            else:
                self.view.draw_frame_only()

        # UI chrome on top
        py5.hint(py5.DISABLE_DEPTH_TEST)
        py5.camera()  # reset to default 2D-friendly camera
        py5.perspective()
        self.draw_header()
        self.draw_dropdowns()
        self.draw_playback_controls()
        self.draw_sound_engine_toggle()
        self.draw_record_button()

    #  UI

    def draw_header(self):
        py5.no_stroke()
        py5.fill(255)
        py5.text_size(36)
        py5.text_align(py5.LEFT, py5.BASELINE)
        py5.text("Sonamic", 20, 50)

        py5.text_size(14)
        py5.text_align(py5.RIGHT, py5.BASELINE)
        py5.fill(180)
        py5.text("Version 1.1", self.width - 20, 44)
        py5.text_align(py5.LEFT, py5.BASELINE)

    def draw_dropdowns(self):
        disabled = self.is_running and not self.is_paused

        py5.text_size(16)
        py5.no_stroke()
        py5.fill(220)
        py5.text("Problem",               self.problem_dropdown_x,   95)
        py5.text("Optimisation Algorithm", self.algorithm_dropdown_x, 95)
        py5.text("Sonification Pathway",   self.pathway_dropdown_x,   95)

        self._draw_single_dropdown(
            self.problem_dropdown_x, self.problem_dropdown_y,
            self.problem_dropdown_w, self.problem_dropdown_h,
            self.problem_options, self.problem_selected_index,
            self.problem_dropdown_open, disabled,
        )
        self._draw_single_dropdown(
            self.algorithm_dropdown_x, self.algorithm_dropdown_y,
            self.algorithm_dropdown_w, self.algorithm_dropdown_h,
            self.algorithm_options, self.algorithm_selected_index,
            self.algorithm_dropdown_open, disabled,
        )
        self._draw_single_dropdown(
            self.pathway_dropdown_x, self.pathway_dropdown_y,
            self.pathway_dropdown_w, self.pathway_dropdown_h,
            self.pathway_options, self.pathway_selected_index,
            self.pathway_dropdown_open, disabled,
        )

    def _draw_single_dropdown(self, x, y, w, h, options, selected_index,
                               is_open, disabled):
        alpha = 100 if disabled else 240

        py5.stroke(alpha)
        py5.stroke_weight(1)
        py5.fill(20)
        py5.rect(x, y, w, h, 4)

        py5.no_stroke()
        py5.text_size(15)
        py5.text_align(py5.LEFT, py5.CENTER)
        if selected_index is not None:
            py5.fill(alpha)
            py5.text(options[selected_index], x + 10, y + h // 2)
        else:
            py5.fill(120 if not disabled else 70)
            py5.text("Select...", x + 10, y + h // 2)

        arrow_x = x + w - 14
        arrow_y = y + h // 2
        py5.fill(alpha)
        py5.triangle(arrow_x - 5, arrow_y - 3,
                     arrow_x + 5, arrow_y - 3,
                     arrow_x,     arrow_y + 4)

        if is_open and not disabled:
            for i, option in enumerate(options):
                opt_y = y + h + i * h
                py5.fill(60 if i == selected_index else 20)
                py5.stroke(80)
                py5.stroke_weight(1)
                py5.rect(x, opt_y, w, h, 0)
                py5.no_stroke()
                py5.fill(240)
                py5.text_align(py5.LEFT, py5.CENTER)
                py5.text_size(15)
                py5.text(option, x + 10, opt_y + h // 2)

        py5.text_align(py5.LEFT, py5.BASELINE)

    def _draw_central_panel_border(self):

        py5.no_fill()
        py5.stroke(80)
        py5.stroke_weight(1)
        py5.rect(self.panel_x, self.panel_y,
                 self.panel_w, self.panel_h, 6)

        # Placeholder text when no view is loaded
        if self.view is None:
            py5.no_stroke()
            py5.fill(100)
            py5.text_size(17)
            py5.text_align(py5.CENTER, py5.CENTER)
            py5.text(
                "Please select a test problem, optimisation algorithm and sonification pathway",
                self.panel_x + self.panel_w // 2,
                self.panel_y + self.panel_h // 2,
            )
            py5.text_align(py5.LEFT, py5.BASELINE)

    def draw_playback_controls(self):
        y  = self.height - 55
        cx = self.width  // 2

        # Play / Pause circle
        py5.stroke(200)
        py5.stroke_weight(1)
        py5.no_fill()
        py5.ellipse(cx - 40, y, 38, 38)

        py5.no_stroke()
        if self.is_running and not self.is_paused:
            # Pause bars
            py5.fill(255)
            py5.rect(cx - 48, y - 9,  5, 18, 1)
            py5.rect(cx - 38, y - 9,  5, 18, 1)
        else:
            # Play triangle
            py5.fill(255)
            py5.triangle(cx - 46, y - 10, cx - 46, y + 10, cx - 28, y)

        # Stop square
        py5.stroke(200)
        py5.stroke_weight(1)
        py5.no_fill()
        py5.rect(cx + 12, y - 16, 32, 32, 3)
        py5.no_stroke()
        py5.fill(255)
        py5.rect(cx + 19, y - 9, 18, 18, 2)
        py5.stroke_weight(1)

    def draw_sound_engine_toggle(self):
        """
        Draw Py/OSC toggle.  non-interactive while running.
        """
        x, y, w, h = self.toggle_x, self.toggle_y, self.toggle_w, self.toggle_h
        half = w // 2
        disabled = self.is_running   # greyed out when running

        py5.no_stroke()
        py5.fill(80 if disabled else 180)
        py5.text_size(12)
        py5.text_align(py5.LEFT, py5.BASELINE)
        py5.text("Sound Engine", x, y - 6)

        for i, label in enumerate(['Py', 'OSC']):
            bx = x + i * half
            is_active = (self.sound_engine == label)

            if disabled:
                # Greyed-out appearance
                py5.stroke(60)
                py5.stroke_weight(1)
                py5.fill(40 if is_active else 15)
                py5.rect(bx, y, half, h, 3)
                py5.no_stroke()
                py5.fill(80 if is_active else 50)
            else:
                # Normal appearance
                py5.stroke(150)
                py5.stroke_weight(1)
                py5.fill(255 if is_active else 15)
                py5.rect(bx, y, half, h, 3)
                py5.no_stroke()
                py5.fill(0 if is_active else 140)

            py5.text_size(12)
            py5.text_align(py5.CENTER, py5.CENTER)
            py5.text(label, bx + half // 2, y + h // 2)

        py5.text_align(py5.LEFT, py5.BASELINE)

    def draw_record_button(self):
        """
        Solid red circle when recording, outlined red circle when idle.
        """
        cx = self.record_btn_x
        cy = self.record_btn_y
        r  = self.record_btn_r

        if self.is_recording:
            # Pulsing filled red circle
            pulse = 0.6 + 0.4 * abs(
                (py5.frame_count % 60) / 30.0 - 1.0)
            py5.no_stroke()
            py5.fill(220, 30, 30, int(255 * pulse))
            py5.ellipse(cx, cy, r * 2, r * 2)
            # "REC" label
            py5.fill(255, 80, 80)
            py5.text_size(10)
            py5.text_align(py5.CENTER, py5.TOP)
            py5.text("REC", cx, cy + r + 4)
        else:
            # Outlined red circle
            py5.stroke(180, 50, 50)
            py5.stroke_weight(2)
            py5.no_fill()
            py5.ellipse(cx, cy, r * 2, r * 2)
            # Inner filled red dot
            py5.no_stroke()
            py5.fill(180, 50, 50)
            py5.ellipse(cx, cy, r * 0.9, r * 0.9)
            # label
            py5.fill(140, 50, 50)
            py5.text_size(10)
            py5.text_align(py5.CENTER, py5.TOP)
            py5.text("REC", cx, cy + r + 4)

        py5.text_align(py5.LEFT, py5.BASELINE)
        py5.stroke_weight(1)

    #  input

    def ui_interaction(self):
        mx, my   = py5.mouse_x, py5.mouse_y
        disabled = self.is_running and not self.is_paused

        # Forward to view param editor (allowed while running)
        if self.view is not None:
            self.view.mouse_pressed(mx, my)

        # Sound engine toggle
        if not self.is_running:
            if (self.toggle_x <= mx <= self.toggle_x + self.toggle_w and
                    self.toggle_y <= my <= self.toggle_y + self.toggle_h):
                half = self.toggle_w // 2
                new_engine = 'Py' if mx < self.toggle_x + half else 'OSC'
                if new_engine != self.sound_engine:
                    self.sound_engine = new_engine
                    self._try_load_view()
                return

        # Problem dropdown header
        if (self._in_rect(mx, my,
                          self.problem_dropdown_x, self.problem_dropdown_y,
                          self.problem_dropdown_w, self.problem_dropdown_h)
                and not disabled):
            self.algorithm_dropdown_open = False
            self.pathway_dropdown_open   = False
            self.problem_dropdown_open   = not self.problem_dropdown_open
            return

        # Problem dropdown options
        if self.problem_dropdown_open and not disabled:
            idx = self._dropdown_option_hit(
                mx, my,
                self.problem_dropdown_x, self.problem_dropdown_y,
                self.problem_dropdown_w, self.problem_dropdown_h,
                len(self.problem_options),
            )
            if idx is not None:
                self.problem_selected_index = idx
                self.problem_dropdown_open  = False
                self._try_load_view()
                return

        # Algorithm dropdown header
        if (self._in_rect(mx, my,
                          self.algorithm_dropdown_x, self.algorithm_dropdown_y,
                          self.algorithm_dropdown_w, self.algorithm_dropdown_h)
                and not disabled):
            self.problem_dropdown_open    = False
            self.pathway_dropdown_open    = False
            self.algorithm_dropdown_open  = not self.algorithm_dropdown_open
            return

        # Algorithm dropdown options
        if self.algorithm_dropdown_open and not disabled:
            idx = self._dropdown_option_hit(
                mx, my,
                self.algorithm_dropdown_x, self.algorithm_dropdown_y,
                self.algorithm_dropdown_w, self.algorithm_dropdown_h,
                len(self.algorithm_options),
            )
            if idx is not None:
                self.algorithm_selected_index = idx
                self.algorithm_dropdown_open  = False
                self._try_load_view()
                return

        # Pathway dropdown header
        if (self._in_rect(mx, my,
                          self.pathway_dropdown_x, self.pathway_dropdown_y,
                          self.pathway_dropdown_w, self.pathway_dropdown_h)
                and not disabled):
            self.problem_dropdown_open    = False
            self.algorithm_dropdown_open  = False
            self.pathway_dropdown_open    = not self.pathway_dropdown_open
            return

        # Pathway dropdown options
        if self.pathway_dropdown_open and not disabled:
            idx = self._dropdown_option_hit(
                mx, my,
                self.pathway_dropdown_x, self.pathway_dropdown_y,
                self.pathway_dropdown_w, self.pathway_dropdown_h,
                len(self.pathway_options),
            )
            if idx is not None:
                self.pathway_selected_index = idx
                self.pathway_dropdown_open  = False
                self._try_load_view()
                return

        # Close all dropdowns on click outside
        if (self.problem_dropdown_open or self.algorithm_dropdown_open
                or self.pathway_dropdown_open):
            self.problem_dropdown_open   = False
            self.algorithm_dropdown_open = False
            self.pathway_dropdown_open   = False
            return

        # Record button (bottom-right)
        if py5.dist(mx, my, self.record_btn_x, self.record_btn_y) < self.record_btn_r + 4:
            self._toggle_recording()
            return

        # Play / Pause circle
        cx = self.width  // 2
        y  = self.height - 55
        if py5.dist(mx, my, cx - 40, y) < 19:
            if not self.is_running:
                self._start()
            else:
                self.is_paused = not self.is_paused
            return

        # Stop square
        if cx + 12 <= mx <= cx + 44 and y - 16 <= my <= y + 16:
            self._stop()
            return

    def key_pressed(self):
        if self.view is not None:
            self.view.key_pressed(py5.key, py5.key_code)

    def _in_rect(self, mx, my, x, y, w, h):
        return x <= mx <= x + w and y <= my <= y + h

    def _dropdown_option_hit(self, mx, my, x, y, w, h, num_options):
        for i in range(num_options):
            opt_y = y + h + i * h
            if self._in_rect(mx, my, x, opt_y, w, h):
                return i
        return None


    #  central panel view

    def _try_load_view(self):
        if (self.problem_selected_index   is None or
                self.algorithm_selected_index is None or
                self.pathway_selected_index   is None):
            return

        problem_name = self.problem_options[self.problem_selected_index]
        algorithm    = self.algorithm_options[self.algorithm_selected_index]
        pathway_key  = self._pathway_keys[
            self.pathway_options[self.pathway_selected_index]
        ]

        # Map display name → problem class
        _problem_map = {
            "DynamicOneMax": DynamicOneMax,
            "RandomIntStr":  RandomIntStr,
            "AllToNone":     AllToNone,
        }
        problem_class = _problem_map.get(problem_name)
        if problem_class is None:
            print(f"Unknown problem: {problem_name}")
            self.view = None
            return

        # Stop existing pathway threads before replacing the view
        if self.view is not None:
            self.view._stop_audio()

        # Silence existing audio
        self.audio_engine.state.clear_all()

        if algorithm == "[1+1] Evolutionary Algorithm":
            self.view = AudioVisualiseOPO(
                problem_class, OPO,
                sound_engine=self.sound_engine,
                audio_engine=self.audio_engine,
                r=2, n=10, p=0.1, q=0.023, bit_shift=1,
                pathway=pathway_key,
            )
        elif algorithm == "Particle Swarm Optimisation":
            self.view = AudioVisualisePSO(
                problem_class, PSO,
                sound_engine=self.sound_engine,
                audio_engine=self.audio_engine,
                num_particles=8, n=3, r=100,
                w=0.5, c1=1.8, c2=1.8, q=0.1, bit_shift=10,
                pathway=pathway_key,
            )
        else:
            print(f"No view for: {problem_name} + {algorithm}")
            self.view = None
            return

        self.view.visuals.set_panel(
            self.panel_x + 4,
            self.panel_y + 4,
            self.panel_w - 8,
            self.panel_h - 8,
        )

    #  playback buttons

    def _start(self):
        if self.view is None:
            print("Select a problem, algorithm and sonification pathway first.")
            return
        self.is_running = True
        self.is_paused  = False
        self.problem_dropdown_open   = False
        self.algorithm_dropdown_open = False
        self.pathway_dropdown_open   = False

    def _stop(self):
        self.is_running = False
        self.is_paused  = False
        self.audio_engine.state.clear_all()
        # Stop recording (exports PNG if active)
        if self.is_recording:
            self._stop_recording()
        # Capture current editor params before view is destroyed
        if self.view is not None:
            saved_r         = self.view.visuals.r
            saved_n         = self.view.visuals.n
            saved_q         = self.view.visuals.q
            saved_bit_shift = self.view.visuals.bit_shift
            is_opo          = isinstance(self.view, AudioVisualiseOPO)
            is_pso          = isinstance(self.view, AudioVisualisePSO)
            if is_opo:
                saved_p = self.view.visuals.p
            if is_pso:
                saved_np = self.view.visuals.num_particles
                saved_w  = self.view.visuals.w
                saved_c1 = self.view.visuals.c1
                saved_c2 = self.view.visuals.c2
        else:
            saved_r = saved_n = None
            is_opo = is_pso = False
        self._try_load_view()
        # Restore saved params into the fresh view
        if saved_r is not None and self.view is not None:
            self.view.visuals.r           = saved_r
            self.view.visuals._r_buf      = str(saved_r)
            self.view.visuals.n           = saved_n
            self.view.visuals._n_buf      = str(saved_n)
            self.view.visuals.q           = saved_q
            self.view.visuals._q_buf      = f"{saved_q:.6f}"
            self.view.visuals.bit_shift   = saved_bit_shift
            self.view.visuals._bit_shift_buf = str(saved_bit_shift)
            if is_opo:
                self.view.visuals.p = saved_p
                self.view.visuals._p_buf = f"{saved_p:.4f}"
            if is_pso:
                self.view.visuals.num_particles = saved_np
                self.view.visuals._np_buf       = str(saved_np)
                self.view.visuals.w             = saved_w
                self.view.visuals._w_buf        = f"{saved_w:.4f}"
                self.view.visuals.c1            = saved_c1
                self.view.visuals._c1_buf       = f"{saved_c1:.4f}"
                self.view.visuals.c2            = saved_c2
                self.view.visuals._c2_buf       = f"{saved_c2:.4f}"
            self.view.visuals._apply_params()


    #  recording

    def _toggle_recording(self):
        """Toggle recording on/off.  Starting requires a loaded view."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _get_current_params(self):
        """Extract display parameters from the active view for the PNG header."""
        if self.view is None:
            return {}
        v = self.view.visuals
        params = {'r': v.r, 'n': v.n}
        if isinstance(self.view, AudioVisualiseOPO):
            params['p'] = f'{v.p:.4f}'
            if v.show_droste_params:
                params['q']         = f'{v.q:.6f}'
                params['bit_shift'] = v.bit_shift
        else:   # PSO
            params['particles'] = v.num_particles
            params['w']         = f'{v.w:.4f}'
            params['c1']        = f'{v.c1:.4f}'
            params['c2']        = f'{v.c2:.4f}'
            if v.show_droste_params:
                params['q']         = f'{v.q:.6f}'
                params['bit_shift'] = v.bit_shift
        return params

    def _start_recording(self):
        """Begin audio recording — only allowed when a view is loaded."""
        if self.view is None:
            print("Select a problem and algorithm before recording.")
            return

        alg_type   = 'OPO' if isinstance(self.view, AudioVisualiseOPO) else 'PSO'

        self.recorder = ExtractAnalysis(alg_type)
        self.recorder.set_params(self._get_current_params())
        self.recorder.set_audio_engine(self.view.engine)

        self.recorder.show_optimum_lines = (self.view._problem_class is not DynamicOneMax)

        self.recorder.start_recording()

        self.view.recorder = self.recorder
        self.is_recording  = True
        print(f"[Recording] Started ({alg_type})")

    def _stop_recording(self):
        """Stop audio capture and export MP3 + PNG."""
        if not self.is_recording or self.recorder is None:
            self.is_recording = False
            return

        # Detach recorder from view so no more steps are appended
        if self.view is not None:
            self.view.recorder = None

        ts         = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        alg        = self.recorder.algorithm_type
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base       = os.path.join(script_dir, f'DynSonOpt_Analysis_{alg}_{ts}')

        n_steps = len(self.recorder.steps)
        self.recorder.stop_and_export(base + '.mp3', base + '.png')
        print(f"[Recording] Stopped — {n_steps} iterations  →  "
              f"{alg}_{ts}.mp3 / .png")

        self.recorder     = None
        self.is_recording = False


### Run!

app = DynamicSonOptApp()

py5.run_sketch(
    block=True,
    sketch_functions={
        "settings":      app.settings,
        "setup":         app.setup,
        "draw":          app.draw,
        "mouse_pressed": app.ui_interaction,
        "key_pressed":   app.key_pressed,
    },
)
