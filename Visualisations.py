"""
Visual display classes for optimisation algorithms.

VisualiseOPO  — (1+1) EA: stats (left), fitness graph (right), rects (centre)
VisualisePSO  — PSO: stats (left), gbest fitness graph (right), 2D/3D/parallel (centre)
"""

import py5
import numpy as np
import time
from collections import deque
from DynamicProblems import DynamicOneMax


def _draw_text_field(x, y, w, h, text, focused):
    border = 200 if focused else 80
    bg     =  30 if focused else 15
    py5.fill(bg)
    py5.stroke(border)
    py5.stroke_weight(1)
    py5.rect(x, y, w, h, 3)
    py5.no_stroke()
    py5.fill(240)
    py5.text_align(py5.LEFT, py5.CENTER)
    cursor = '|' if focused and (py5.frame_count // 30) % 2 == 0 else ''
    py5.text(text + cursor, x + 5, y + h // 2)
    py5.text_align(py5.LEFT, py5.BASELINE)


def _draw_dropdown_header(x, y, w, h, current_value, is_open):
    py5.fill(20)
    py5.stroke(200 if is_open else 80)
    py5.stroke_weight(1)
    py5.rect(x, y, w, h, 3)
    py5.no_stroke()
    py5.fill(240)
    py5.text_align(py5.LEFT, py5.CENTER)
    py5.text(current_value, x + 5, y + h // 2)
    ax, ay = x + w - 10, y + h // 2
    py5.fill(200)
    if is_open:
        py5.triangle(ax - 4, ay + 3, ax + 4, ay + 3, ax, ay - 3)
    else:
        py5.triangle(ax - 4, ay - 3, ax + 4, ay - 3, ax, ay + 3)
    py5.text_align(py5.LEFT, py5.BASELINE)


def _in_rect(mx, my, x, y, w, h):
    return x <= mx <= x + w and y <= my <= y + h

#  Visualiser for (1+1) EA

class VisualiseOPO:
    """
    Visual display for the (1+1) EA.

    Editable parameters (always shown):            r, n, p
    Editable parameters (DynamicOneMax):  q, bit_shift
    Movement type is fixed by the problem class and shown read-only.
    """

    def __init__(self, problem, opt_alg,
                 r=2, n=10, p=0.1, q=0.023, bit_shift=1,
                 rate=100, wait_time=0.001):

        self.r             = r
        self.n             = n
        self.rate          = rate
        self.p             = p if p is not None else 0.1
        self.q             = q
        self.bit_shift     = bit_shift
        self.wait_time     = wait_time

        # Show q / bit_shift fields only for DynamicOneMax
        self.show_droste_params = (problem is DynamicOneMax)

        self._problem_class = problem
        self._opt_alg_class = opt_alg

        self.native_width  = 1280
        self.native_height = 720
        self.panel_x  = 0
        self.panel_y  = 0
        self.panel_w  = self.native_width
        self.panel_h  = self.native_height
        self._embedded = False

        self._recalc_layout()

        self.max_history      = 50
        self.distance_history = deque(maxlen=self.max_history)

        # param editor state
        self._field_focus   = None
        self._error_msg     = ''
        self._error_timer   = 0

        self._r_buf         = str(r)
        self._n_buf         = str(n)
        self._p_buf         = f"{self.p:.4f}"
        self._q_buf         = f"{self.q:.6f}"
        self._bit_shift_buf = str(bit_shift)

        # hit-rects (None until drawn)
        self._r_box         = None
        self._n_box         = None
        self._p_box         = None
        self._q_box         = None
        self._bit_shift_box = None

        # Signature: on_params_changed(r, n, p, q, bit_shift)
        self.on_params_changed = None

        # Randomised dummies so rectangles are visible before play is pressed
        rng = np.random.default_rng()
        self._dummy_candidate = rng.integers(0, max(r, 2), size=n)
        self._dummy_optimum   = rng.integers(0, max(r, 2), size=n)

    # layout

    def set_panel(self, x, y, w, h):
        self.panel_x   = x
        self.panel_y   = y
        self.panel_w   = w
        self.panel_h   = h
        self._embedded = True
        self._recalc_layout()

    def _recalc_layout(self):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y
        sw     = W // 3
        self.rect_width       = max(20, sw // 3)
        self.rect_height      = H * 0.75
        self.rect_spacing     = max(5, sw // 9)
        self.candidate_rect_x = ox + sw + self.rect_spacing
        self.optimum_rect_x   = self.candidate_rect_x + self.rect_width + self.rect_spacing
        self.rect_y           = oy + (H - self.rect_height) // 2

    # param apply

    def _apply_params(self):
        """Reset visual state and notify owner. Does NOT rebuild opt_alg/problem."""
        self._recalc_layout()
        self.distance_history.clear()
        rng = np.random.default_rng()
        self._dummy_candidate = rng.integers(0, max(self.r, 2), size=self.n)
        self._dummy_optimum   = rng.integers(0, max(self.r, 2), size=self.n)
        if self.on_params_changed is not None:
            self.on_params_changed(self.r, self.n, self.p, self.q,
                                   self.bit_shift)

    # py5 standalone use

    def settings(self):
        py5.size(self.native_width, self.native_height)

    def setup(self):
        self.font = py5.create_font("Courier New", 20)
        py5.text_font(self.font)

    # rectangle strip drawing
    def rectangles(self, x, y, width, height, values):
        if len(values) == 0:
            return
        strip_height = height / len(values)
        r_minus_1    = max(self.r - 1, 1)
        for i, value in enumerate(values):
            strip_y  = y + i * strip_height
            grayness = int((value / r_minus_1) * 255)
            py5.fill(grayness)
            py5.stroke(grayness)
            py5.rect(x, strip_y, width, strip_height)

    # stats + param editor

    def _draw_params(self, x, y, dy, fh, is_running):
        field_w = int((self.panel_w // 3) * 0.58)
        label_w = max(28, int(dy * 0.55))

        py5.text_size(max(15, int(fh)))
        py5.no_stroke()

        # r
        py5.fill(220); py5.text_align(py5.LEFT, py5.BASELINE)
        py5.text("r:", x, y + fh - 3)
        self._r_box = (x + label_w + 4, y, field_w, fh)
        _draw_text_field(x + label_w + 4, y, field_w, fh,
                         self._r_buf, self._field_focus == 'r')
        y += dy

        # n
        py5.fill(220); py5.text_align(py5.LEFT, py5.BASELINE)
        py5.text("n:", x, y + fh - 3)
        self._n_box = (x + label_w + 4, y, field_w, fh)
        _draw_text_field(x + label_w + 4, y, field_w, fh,
                         self._n_buf, self._field_focus == 'n')
        y += dy

        # p
        py5.fill(220); py5.text_align(py5.LEFT, py5.BASELINE)
        py5.text("p:", x, y + fh - 3)
        self._p_box = (x + label_w + 4, y, field_w, fh)
        _draw_text_field(x + label_w + 4, y, field_w, fh,
                         self._p_buf, self._field_focus == 'p')

        # q and bit_shift — DynamicOneMax only
        if self.show_droste_params:
            y += dy
            py5.fill(220); py5.text_align(py5.LEFT, py5.BASELINE)
            py5.text("q:", x, y + fh - 3)
            self._q_box = (x + label_w + 4, y, field_w, fh)
            _draw_text_field(x + label_w + 4, y, field_w, fh,
                             self._q_buf, self._field_focus == 'q')
            y += dy
            py5.fill(220); py5.text_align(py5.LEFT, py5.BASELINE)
            py5.text("bs:", x, y + fh - 3)
            self._bit_shift_box = (x + label_w + 4, y, field_w, fh)
            _draw_text_field(x + label_w + 4, y, field_w, fh,
                             self._bit_shift_buf,
                             self._field_focus == 'bit_shift')
        else:
            self._q_box         = None
            self._bit_shift_box = None

        py5.text_align(py5.LEFT, py5.BASELINE)

        if self._error_timer > 0:
            self._error_timer -= 1
            py5.fill(220, 60, 60); py5.no_stroke()
            py5.text_size(max(11, int(fh * 0.70)))
            py5.text(self._error_msg, x, y + fh + dy)

    def stats(self, iteration, current_fitness, is_running):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y

        py5.no_stroke(); py5.fill(240)
        ts = max(17, int(H * 0.044))
        py5.text_size(ts)
        py5.text_align(py5.LEFT, py5.BASELINE)
        x  = ox + max(6, (W // 3) // 9)
        y  = oy + int((H - H * 0.75) // 2)
        dy = H * 0.065
        fh = max(20, int(dy * 0.62))

        py5.text(f"Iteration: {iteration}",             x, y); y += dy
        py5.text(f"Current Fitness: {current_fitness}", x, y); y += dy

        sep_y = int(y + dy * 0.5)
        py5.stroke(80); py5.stroke_weight(1)
        py5.line(x, sep_y, ox + W // 3 - max(6, (W // 3) // 9), sep_y)
        py5.no_stroke()
        y = sep_y + dy * 0.4

        self._draw_params(x, y, dy, fh, is_running)

    # fitness history graph

    def fitness_graph(self):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y
        sw     = W // 3

        graph_x = ox + sw * 2.125
        graph_y = oy + int((H - H * 0.75) // 2)
        graph_w = sw * 0.8
        graph_h = H * 0.75

        py5.no_fill(); py5.stroke(200)
        py5.rect(graph_x, graph_y, graph_w, graph_h)

        if len(self.distance_history) < 2:
            return

        max_fitness = max((self.r * self.n)/ 2, 1)
        margin      = graph_h * 0.05

        py5.no_fill(); py5.stroke(200)
        py5.begin_shape()
        for i, d in enumerate(self.distance_history):
            gx = graph_x + (i / (self.max_history - 1)) * graph_w
            gy = graph_y + graph_h - margin - (d / max_fitness) * (graph_h - 2 * margin)
            py5.vertex(gx, gy)
        py5.end_shape()

        py5.text_size(15); py5.fill(200)
        for i in range(5):
            frac   = i / 4
            val    = int(frac * max_fitness)
            tick_y = graph_y + graph_h - margin - frac * (graph_h - 2 * margin)
            py5.stroke(120); py5.line(graph_x - 6, tick_y, graph_x, tick_y)
            py5.fill(200); py5.no_stroke()
            py5.text_align(py5.RIGHT, py5.CENTER)
            py5.text(str(val), graph_x - 10, tick_y)

        py5.text_align(py5.LEFT, py5.BASELINE)
        py5.fill(240); py5.text_size(17)
        py5.text("Distance to optimum (last 50 iters)", graph_x, graph_y - 14)

    # full frame

    def draw_frame(self, current_candidate, optimum, iteration,
                   current_fitness, is_running=True):
        if not self._embedded:
            py5.background(0)

        py5.hint(py5.DISABLE_DEPTH_TEST)

        self.stats(iteration, current_fitness, is_running)
        self.fitness_graph()

        self.rectangles(self.candidate_rect_x, self.rect_y,
                        self.rect_width, self.rect_height, current_candidate)
        self.rectangles(self.optimum_rect_x, self.rect_y,
                        self.rect_width, self.rect_height, optimum)

        py5.fill(240); py5.text_size(18); py5.text_align(py5.CENTER, py5.BASELINE)
        py5.text("Candidate",
                 self.candidate_rect_x + self.rect_width / 2, self.rect_y - 14)
        py5.text("Optimum",
                 self.optimum_rect_x + self.rect_width / 2, self.rect_y - 14)
        py5.text_align(py5.LEFT, py5.BASELINE)

    def draw_idle(self):
        """Idle frame — randomised dummy data so rectangles are always visible."""
        self.draw_frame(self._dummy_candidate, self._dummy_optimum,
                        0, 0, is_running=False)

    # input handling

    def mouse_pressed(self, mx, my, is_running=False):
        for key, box in [('r',         self._r_box),
                         ('n',         self._n_box),
                         ('p',         self._p_box),
                         ('q',         self._q_box),
                         ('bit_shift', self._bit_shift_box)]:
            if box and _in_rect(mx, my, *box):
                if self._field_focus is not None and self._field_focus != key:
                    self._commit()
                self._field_focus = key
                return

        if self._field_focus is not None:
            self._commit()
            self._field_focus = None

    def key_pressed(self, key, key_code, is_running=False):
        if self._field_focus is None:
            return
        focus   = self._field_focus
        buf_map = {'r': '_r_buf', 'n': '_n_buf',
                   'p': '_p_buf', 'q': '_q_buf',
                   'bit_shift': '_bit_shift_buf'}
        attr = buf_map.get(focus)
        if attr is None:
            return

        if key in ('\n', '\r'):
            self._commit(); self._field_focus = None; return
        if key == '\x1b':
            self._r_buf         = str(self.r)
            self._n_buf         = str(self.n)
            self._p_buf         = f"{self.p:.4f}"
            self._q_buf         = f"{self.q:.6f}"
            self._bit_shift_buf = str(self.bit_shift)
            self._field_focus = None; return
        if key == '\t':
            self._commit()
            order = ['r', 'n', 'p']
            if self.show_droste_params:
                order += ['q', 'bit_shift']
            idx = order.index(focus) if focus in order else 0
            self._field_focus = order[(idx + 1) % len(order)]; return
        if key == '\x08':
            setattr(self, attr, getattr(self, attr)[:-1]); return
        if key == '.' and focus in ('p', 'q'):
            buf = getattr(self, attr)
            if '.' not in buf:
                setattr(self, attr, buf + key)
            return
        if key.isdigit():
            setattr(self, attr, getattr(self, attr) + key); return

    # validation / commit

    def _commit(self):
        ok = True

        try:
            v = int(self._r_buf)
            if v < 2: raise ValueError
            self.r = v
        except ValueError:
            self._error_msg   = f"r must be integer >= 2  (got '{self._r_buf}')"
            self._error_timer = 120
            self._r_buf = str(self.r); ok = False

        try:
            v = int(self._n_buf)
            if v < 1: raise ValueError
            self.n = v
        except ValueError:
            if ok:
                self._error_msg   = f"n must be integer >= 1  (got '{self._n_buf}')"
                self._error_timer = 120
            self._n_buf = str(self.n); ok = False

        try:
            v = float(self._p_buf)
            if not (0.0 < v <= 1.0): raise ValueError
            self.p = v
        except ValueError:
            if ok:
                self._error_msg   = f"p must be float in (0,1]  (got '{self._p_buf}')"
                self._error_timer = 120
            self._p_buf = f"{self.p:.4f}"; ok = False

        if self.show_droste_params:
            try:
                v = float(self._q_buf)
                if v <= 0: raise ValueError
                self.q = v
            except ValueError:
                if ok:
                    self._error_msg   = f"q must be float > 0  (got '{self._q_buf}')"
                    self._error_timer = 120
                self._q_buf = f"{self.q:.6f}"; ok = False

            try:
                v = int(self._bit_shift_buf)
                if v < 1: raise ValueError
                self.bit_shift = v
            except ValueError:
                if ok:
                    self._error_msg   = f"bit_shift must be integer >= 1"
                    self._error_timer = 120
                self._bit_shift_buf = str(self.bit_shift); ok = False

        if ok:
            self._apply_params()
        return ok

    # standalone run

    def run(self):
        from DynamicOptimisers import OPO
        self._sa_opt  = OPO(r=self.r, n=self.n, p=self.p)
        if self._problem_class is DynamicOneMax:
            self._sa_prob = self._problem_class(r=self.r, n=self.n, q=self.q, bit_shift=self.bit_shift)
        else:
            self._sa_prob = self._problem_class(r=self.r, n=self.n, rate=self.rate)
        self._sa_gen  = self._sa_prob.run_problem()

        def _draw():
            ps = next(self._sa_gen)
            (cand, fit, _, _, _, _) = self._sa_opt.iterate_candidate(ps['optimum'])
            self.distance_history.append(fit)
            self.draw_frame(cand, ps['optimum'], ps['iteration'], fit,
                            is_running=True)
            time.sleep(self.wait_time)

        py5.run_sketch(block=False, sketch_functions={
            'settings': self.settings, 'setup': self.setup, 'draw': _draw,
        })

#  Visualiser for PSO


class VisualisePSO:

    def __init__(self, problem, opt_alg,
                 num_particles=8, n=3, r=100, rate=100, wait_time=0.001,
                 w=0.5, c1=1.8, c2=1.8, q=0.1, bit_shift=10):

        self.num_particles = num_particles
        self.n             = n
        self.r             = r
        self.rate          = rate
        self.wait_time     = wait_time

        # PSO algorithm params
        self.w  = w
        self.c1 = c1
        self.c2 = c2

        # droste problem param
        self.q         = q
        self.bit_shift = bit_shift

        # Show q / bit_shift fields only for DynamicOneMax (droste drift)
        self.show_droste_params = (problem is DynamicOneMax)

        self._problem_class = problem
        self._opt_alg_class = opt_alg

        self.native_width  = 1280
        self.native_height = 720
        self.panel_x  = 0
        self.panel_y  = 0
        self.panel_w  = self.native_width
        self.panel_h  = self.native_height
        self._embedded = False

        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.box_size   = 200

        self.trail_history = [deque(maxlen=12) for _ in range(self.num_particles)]
        self.gbest_history = deque(maxlen=50)

        # inline param editor state
        self._field_focus   = None
        self._error_msg     = ''
        self._error_timer   = 0

        self._r_buf         = str(r)
        self._n_buf         = str(n)
        self._np_buf        = str(num_particles)
        self._w_buf         = f"{self.w:.4f}"
        self._c1_buf        = f"{self.c1:.4f}"
        self._c2_buf        = f"{self.c2:.4f}"
        self._q_buf         = f"{self.q:.6f}"
        self._bit_shift_buf = str(bit_shift)

        # hit-rects (None until drawn)
        self._r_box         = None
        self._n_box         = None
        self._np_box        = None
        self._w_box         = None
        self._c1_box        = None
        self._c2_box        = None
        self._q_box         = None
        self._bit_shift_box = None

        # Signature: on_params_changed(r, n, num_particles, w, c1, c2, q, bit_shift)
        self.on_params_changed = None

        self._dummy_positions = np.full((num_particles, max(n, 3)), r / 2.0)
        self._dummy_optimum   = np.full(max(n, 3), r / 2.0)
        self._dummy_gbest     = np.full(max(n, 3), r / 2.0)

    # layout

    def set_panel(self, x, y, w, h):
        self.panel_x   = x
        self.panel_y   = y
        self.panel_w   = w
        self.panel_h   = h
        self._embedded = True
        self.box_size  = int(min(w // 3, h) * 0.38)

    # param apply (visual side only)

    def _apply_params(self):
        """Reset visual state and notify owner. Does NOT rebuild opt_alg/problem."""
        self.gbest_history.clear()
        for t in self.trail_history:
            t.clear()
        self.trail_history = [deque(maxlen=12) for _ in range(self.num_particles)]
        self._dummy_positions = np.full(
            (self.num_particles, max(self.n, 3)), self.r / 2.0)
        self._dummy_optimum = np.full(max(self.n, 3), self.r / 2.0)
        self._dummy_gbest   = np.full(max(self.n, 3), self.r / 2.0)

        if self.on_params_changed is not None:
            self.on_params_changed(self.r, self.n,
                                   self.num_particles,
                                   self.w, self.c1, self.c2,
                                   self.q, self.bit_shift)

    # py5 lifecycle (standalone use)

    def settings(self):
        py5.size(self.native_width, self.native_height, py5.P3D)

    def setup(self):
        self.font = py5.create_font("Courier New", 20)
        py5.text_font(self.font)

    # coordinate mapping

    def map_pos_3d(self, position):
        scale = (self.box_size / 2) * 0.8
        return tuple((position[i] / self.r - 0.5) * 2 * scale for i in range(3))

    def map_pos_2d(self, position, panel_x, panel_y, panel_w, panel_h):
        margin = 20
        x = panel_x + margin + (position[0] / self.r) * (panel_w - 2 * margin)
        y = panel_y + margin + (1.0 - position[1] / self.r) * (panel_h - 2 * margin)
        return x, y

    # stats + param editor

    def _draw_params(self, x, y, dy, fh, is_running):
        field_w = int((self.panel_w // 3) * 0.58)
        label_w = max(28, int(dy * 0.55))

        py5.text_size(max(15, int(fh)))
        py5.no_stroke()

        # helper: draw one labelled text field and advance y
        def field(label, key, buf_attr, box_attr, yy):
            py5.fill(220); py5.text_align(py5.LEFT, py5.BASELINE)
            py5.text(label, x, yy + fh - 3)
            fx  = x + label_w + 4
            box = (fx, yy, field_w, fh)
            setattr(self, box_attr, box)
            _draw_text_field(fx, yy, field_w, fh,
                             getattr(self, buf_attr),
                             self._field_focus == key)
            return yy + dy

        y = field("r:",    'r',  '_r_buf',  '_r_box',  y)
        y = field("n:",    'n',  '_n_buf',  '_n_box',  y)
        y = field("parts:", 'np', '_np_buf', '_np_box', y)
        y = field("w:",    'w',  '_w_buf',  '_w_box',  y)
        y = field("c1:",   'c1', '_c1_buf', '_c1_box', y)
        y = field("c2:",   'c2', '_c2_buf', '_c2_box', y)

        # q and bit_shift — droste (DynamicOneMax) only
        if self.show_droste_params:
            y = field("q:",   'q',         '_q_buf',         '_q_box',         y)
            y = field("bs:",  'bit_shift', '_bit_shift_buf',  '_bit_shift_box', y)
        else:
            self._q_box         = None
            self._bit_shift_box = None

        py5.text_align(py5.LEFT, py5.BASELINE)

        if self._error_timer > 0:
            self._error_timer -= 1
            py5.fill(220, 60, 60); py5.no_stroke()
            py5.text_size(max(11, int(fh * 0.70)))
            py5.text(self._error_msg, x, y + fh + dy)

    def stats(self, iteration, gbest_value, pbest_values,
              is_running):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y

        py5.no_stroke(); py5.fill(240)
        ts = max(15, int(H * 0.036))
        py5.text_size(ts)
        py5.text_align(py5.LEFT, py5.BASELINE)
        x  = ox + max(6, (W // 3) // 9)
        y  = oy + int((H - H * 0.75) // 2)
        dy = H * 0.062
        fh = max(20, int(dy * 0.62))

        # Only live state above the separator line — no particles count here
        py5.text(f"Iteration: {iteration}",         x, y); y += dy
        py5.text(f"Global Best: {gbest_value:.4f}", x, y); y += dy

        sep_y = int(y + dy * 0.5)
        py5.stroke(80); py5.stroke_weight(1)
        py5.line(x, sep_y, ox + W // 3 - max(6, (W // 3) // 9), sep_y)
        py5.no_stroke()
        y = sep_y + dy * 0.4

        self._draw_params(x, y, dy, fh, is_running)

    # fitness history graph

    def fitness_graph(self):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y
        sw     = W // 3

        graph_x = ox + sw * 2.125
        graph_y = oy + int((H - H * 0.75) // 2)
        graph_w = sw * 0.8
        graph_h = H * 0.75

        py5.no_fill(); py5.stroke(200)
        py5.rect(graph_x, graph_y, graph_w, graph_h)

        if len(self.gbest_history) < 2:
            return

        max_fitness = max(self.r * np.sqrt(self.n), 1)
        margin      = graph_h * 0.05

        py5.no_fill(); py5.stroke(200)
        py5.begin_shape()
        for i, v in enumerate(self.gbest_history):
            gx = graph_x + (i / 49) * graph_w
            gy = graph_y + graph_h - margin - (v / max_fitness) * (graph_h - 2 * margin)
            py5.vertex(gx, gy)
        py5.end_shape()

        py5.text_size(15); py5.fill(200)
        for i in range(5):
            frac   = i / 4
            val    = int(frac * max_fitness)
            tick_y = graph_y + graph_h - margin - frac * (graph_h - 2 * margin)
            py5.stroke(120); py5.line(graph_x - 6, tick_y, graph_x, tick_y)
            py5.fill(200); py5.no_stroke()
            py5.text_align(py5.RIGHT, py5.CENTER)
            py5.text(str(val), graph_x - 10, tick_y)

        py5.text_align(py5.LEFT, py5.BASELINE)
        py5.fill(240); py5.text_size(17)
        py5.text("Global best fitness (last 50 iters)", graph_x, graph_y - 14)

    # 3D particle view

    def _draw_3d(self, gbest_position, positions, optimum):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y
        sw     = W // 3
        cx     = ox + sw + sw // 2
        cy     = oy + int(H * 0.55)

        py5.hint(py5.ENABLE_DEPTH_TEST)

        fov    = py5.PI / 12
        cam_z  = (H / 2.0) / py5.tan(fov / 2.0)
        aspect = float(py5.width) / float(py5.height)
        py5.perspective(fov, aspect, cam_z * 0.01, cam_z * 10.0)
        py5.camera(cx, cy, cam_z, cx, cy, 0, 0, 1, 0)

        py5.push_matrix()
        py5.translate(cx, cy, 0)
        self.rotation_y += 0.003
        py5.rotate_x(self.rotation_x)
        py5.rotate_y(self.rotation_y)

        py5.no_fill(); py5.stroke(255); py5.stroke_weight(0.5)
        py5.box(self.box_size)

        for i in range(self.num_particles):
            trail = self.trail_history[i]
            if len(trail) < 2:
                continue
            for j in range(1, len(trail)):
                alpha = int((j / len(trail)) * 50)
                px1, py1, pz1 = self.map_pos_3d(trail[j - 1])
                px2, py2, pz2 = self.map_pos_3d(trail[j])
                py5.stroke(255, alpha); py5.stroke_weight(0.8)
                py5.line(px1, py1, pz1, px2, py2, pz2)

        for i in range(self.num_particles):
            px, py_, pz = self.map_pos_3d(positions[i])
            py5.no_stroke(); py5.fill(255)
            py5.push_matrix(); py5.translate(px, py_, pz)
            py5.sphere(max(3, self.box_size // 40))
            py5.pop_matrix()

        gx, gy, gz = self.map_pos_3d(gbest_position)
        py5.no_stroke(); py5.fill(255, 200, 0)
        py5.push_matrix(); py5.translate(gx, gy, gz)
        py5.sphere(max(5, self.box_size // 25))
        py5.pop_matrix()

        ox3, oy3, oz3 = self.map_pos_3d(optimum)
        py5.stroke(230, 70); py5.stroke_weight(1)
        py5.line(gx, gy, gz, ox3, oy3, oz3)

        pulse = 1.0 + 0.2 * np.sin(py5.frame_count * 0.1)
        py5.no_stroke(); py5.fill(130, 190)
        py5.push_matrix(); py5.translate(ox3, oy3, oz3)
        py5.sphere(max(10, self.box_size // 15) * pulse)
        py5.pop_matrix()

        py5.pop_matrix()
        py5.camera(); py5.perspective()
        py5.hint(py5.DISABLE_DEPTH_TEST)

        py5.fill(240); py5.text_size(18); py5.text_align(py5.CENTER, py5.BASELINE)
        py5.text("Particles (3D)", cx, oy + int((H - H * 0.75) // 2) - 14)
        py5.text_align(py5.LEFT, py5.BASELINE)

    # 2D particle view

    def _draw_2d(self, gbest_position, positions, optimum):
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y
        sw     = W // 3
        margin = 40

        p2d_x = ox + sw + margin
        p2d_y = oy + int((H - H * 0.75) // 2)
        p2d_w = sw - margin * 2
        p2d_h = int(H * 0.75)

        py5.no_fill(); py5.stroke(255); py5.stroke_weight(0.5)
        py5.rect(p2d_x, p2d_y, p2d_w, p2d_h)

        for i in range(self.num_particles):
            trail = self.trail_history[i]
            if len(trail) < 2:
                continue
            for j in range(1, len(trail)):
                alpha = int((j / len(trail)) * 50)
                x1, y1 = self.map_pos_2d(trail[j-1], p2d_x, p2d_y, p2d_w, p2d_h)
                x2, y2 = self.map_pos_2d(trail[j],   p2d_x, p2d_y, p2d_w, p2d_h)
                py5.stroke(255, alpha); py5.stroke_weight(0.8)
                py5.line(x1, y1, x2, y2)

        for i in range(self.num_particles):
            px, py_ = self.map_pos_2d(positions[i], p2d_x, p2d_y, p2d_w, p2d_h)
            py5.no_stroke(); py5.fill(255); py5.circle(px, py_, 6)

        gx, gy = self.map_pos_2d(gbest_position, p2d_x, p2d_y, p2d_w, p2d_h)
        py5.no_stroke(); py5.fill(255, 200, 0); py5.circle(gx, gy, 9)

        ox2, oy2 = self.map_pos_2d(optimum, p2d_x, p2d_y, p2d_w, p2d_h)
        pulse = 1.0 + 0.2 * np.sin(py5.frame_count * 0.1)
        py5.no_stroke(); py5.fill(130, 190); py5.circle(ox2, oy2, 28 * pulse)

        py5.stroke(230, 70); py5.stroke_weight(1); py5.line(gx, gy, ox2, oy2)

        py5.fill(240); py5.text_size(18); py5.text_align(py5.CENTER, py5.BASELINE)
        py5.text("Particles (2D)", p2d_x + p2d_w / 2, p2d_y - 14)
        py5.text_align(py5.LEFT, py5.BASELINE)

    # parallel-coordinates view (n > 3)

    def _draw_parallel(self, gbest_position, positions, optimum):
        """Parallel-coordinates display for n > 3 dimensions."""
        W, H   = self.panel_w, self.panel_h
        ox, oy = self.panel_x, self.panel_y
        sw     = W // 3
        margin = 20

        # Panel bounds (centre third, same vertical extent as 2D/3D views)
        p_x = ox + sw + margin
        p_y = oy + int((H - H * 0.75) // 2)
        p_w = sw - margin * 2
        p_h = int(H * 0.75)

        n = self.n

        # Vertical layout
        title_gap = 20                     # above panel for heading
        axis_bot  = 46                     # below axis for tick labels + title
        val_top   = p_y + title_gap + 6    # y position that maps to value r
        val_bot   = p_y + p_h - axis_bot - 8   # y position that maps to value 0
        ax_y      = p_y + p_h - axis_bot   # y of the drawn horizontal axis

        # Horizontal positions of the n dimension columns
        h_pad = 18
        ax_x0 = p_x + h_pad
        ax_x1 = p_x + p_w - h_pad
        if n > 1:
            span   = ax_x1 - ax_x0
            dim_xs = [ax_x0 + i * span / (n - 1) for i in range(n)]
        else:
            mid    = (ax_x0 + ax_x1) / 2
            dim_xs = [mid]

        def map_y(val):
            frac = float(np.clip(val, 0, self.r)) / self.r
            return val_bot - frac * (val_bot - val_top)

        def draw_string(pos, cr, cg, cb, alpha, weight, dot_r=5):
            py5.stroke(cr, cg, cb, alpha)
            py5.stroke_weight(weight)
            py5.no_fill()
            for i in range(n - 1):
                py5.line(dim_xs[i],     map_y(pos[i]),
                         dim_xs[i + 1], map_y(pos[i + 1]))
            py5.no_stroke()
            py5.fill(cr, cg, cb, alpha)
            for i in range(n):
                py5.circle(dim_xs[i], map_y(pos[i]), dot_r)

        # 1. Other particles (faint grey)
        if positions is not None:
            for p_pos in positions:
                draw_string(p_pos, 160, 160, 160, 130, 1.0, dot_r=4)

        # 2. Global best (yellow)
        if gbest_position is not None:
            draw_string(gbest_position, 255, 200, 0, 255, 1.8, dot_r=6)

        # 3. Optimum — pulsating glow then solid
        if optimum is not None:
            pulse = 1.0 + 0.3 * np.sin(py5.frame_count * 0.1)
            for glow_w, glow_a in [(12, 18), (7, 32), (3, 55)]:
                py5.stroke(130, 200, 255, int(glow_a * pulse))
                py5.stroke_weight(glow_w)
                py5.no_fill()
                for i in range(n - 1):
                    py5.line(dim_xs[i],     map_y(optimum[i]),
                             dim_xs[i + 1], map_y(optimum[i + 1]))
                py5.no_stroke()
                py5.fill(130, 200, 255, int(glow_a * pulse))
                gr = glow_w * 1.4 * pulse
                for i in range(n):
                    py5.circle(dim_xs[i], map_y(optimum[i]), gr)
            draw_string(optimum, 130, 200, 255, 255, 1.5, dot_r=5)

        # 4. Horizontal axi
        py5.stroke(200); py5.stroke_weight(1.5)
        py5.line(ax_x0, ax_y, ax_x1, ax_y)

        # Tick marks and dimension labels
        ts = max(10, int(p_h * 0.052))
        py5.text_size(ts)
        if n > 1:
            pixel_gap = (ax_x1 - ax_x0) / (n - 1)
        else:
            pixel_gap = ax_x1 - ax_x0
        stride = max(1, int(np.ceil((ts * 1.6) / max(pixel_gap, 1))))

        for i in range(n):
            py5.stroke(200); py5.stroke_weight(1.5)
            py5.line(dim_xs[i], ax_y, dim_xs[i], ax_y + 5)
            if i % stride == 0:
                py5.no_stroke(); py5.fill(200)
                py5.text_align(py5.CENTER, py5.TOP)
                py5.text(str(i + 1), dim_xs[i], ax_y + 7)

        # "Dimension" title below the tick labels
        py5.text_size(max(11, int(p_h * 0.054)))
        py5.fill(200)
        py5.text_align(py5.CENTER, py5.BOTTOM)
        py5.text("Dimension", (ax_x0 + ax_x1) / 2, p_y + p_h - 2)

        # Panel heading
        py5.fill(240); py5.text_size(16)
        py5.text_align(py5.CENTER, py5.BASELINE)
        py5.text("Particles", (ax_x0 + ax_x1) / 2, p_y - 14)
        py5.text_align(py5.LEFT, py5.BASELINE)

    # full frame

    def draw_frame(self, gbest_value, gbest_position, pbest_values,
                   pbest_positions, positions, velocities, position_fits, optimum, iteration, optimum_moved,
                   is_running=True):

        if not self._embedded:
            py5.background(0)

        py5.hint(py5.DISABLE_DEPTH_TEST)

        self.stats(iteration, gbest_value, pbest_values,
                   is_running)
        self.fitness_graph()

        if self.n == 3:
            self._draw_3d(gbest_position, positions, optimum)
        elif self.n == 2:
            self._draw_2d(gbest_position, positions, optimum)
        else:
            self._draw_parallel(gbest_position, positions, optimum)

    def draw_idle(self):
        dummy_pbest = np.zeros(self.num_particles)
        self.draw_frame(
            0.0, self._dummy_gbest, dummy_pbest, None,
            self._dummy_positions, None, dummy_pbest,
            self._dummy_optimum, 0, False,
            is_running=False,
        )

    # input handling

    def mouse_pressed(self, mx, my, is_running=False):
        fields = [('r',         self._r_box),
                  ('n',         self._n_box),
                  ('np',        self._np_box),
                  ('w',         self._w_box),
                  ('c1',        self._c1_box),
                  ('c2',        self._c2_box),
                  ('q',         self._q_box),
                  ('bit_shift', self._bit_shift_box)]
        for key, box in fields:
            if box and _in_rect(mx, my, *box):
                if self._field_focus is not None and self._field_focus != key:
                    self._commit()
                self._field_focus = key
                return

        if self._field_focus is not None:
            self._commit()
            self._field_focus = None

    def key_pressed(self, key, key_code, is_running=False):
        if self._field_focus is None:
            return
        focus   = self._field_focus
        buf_map = {
            'r':         '_r_buf',
            'n':         '_n_buf',
            'np':        '_np_buf',
            'w':         '_w_buf',
            'c1':        '_c1_buf',
            'c2':        '_c2_buf',
            'q':         '_q_buf',
            'bit_shift': '_bit_shift_buf',
        }
        attr = buf_map.get(focus)
        if attr is None:
            return

        float_fields = {'w', 'c1', 'c2', 'q'}

        if key in ('\n', '\r'):
            self._commit(); self._field_focus = None; return
        if key == '\x1b':
            self._r_buf         = str(self.r)
            self._n_buf         = str(self.n)
            self._np_buf        = str(self.num_particles)
            self._w_buf         = f"{self.w:.4f}"
            self._c1_buf        = f"{self.c1:.4f}"
            self._c2_buf        = f"{self.c2:.4f}"
            self._q_buf         = f"{self.q:.6f}"
            self._bit_shift_buf = str(self.bit_shift)
            self._field_focus = None; return
        if key == '\t':
            self._commit()
            order = ['r', 'n', 'np', 'w', 'c1', 'c2']
            if self.show_droste_params:
                order += ['q', 'bit_shift']
            idx = order.index(focus) if focus in order else 0
            self._field_focus = order[(idx + 1) % len(order)]; return
        if key == '\x08':
            setattr(self, attr, getattr(self, attr)[:-1]); return
        if key == '.' and focus in float_fields:
            buf = getattr(self, attr)
            if '.' not in buf:
                setattr(self, attr, buf + key)
            return
        if key == '-' and focus in float_fields:
            buf = getattr(self, attr)
            if not buf:
                setattr(self, attr, '-')
            return
        if key.isdigit():
            setattr(self, attr, getattr(self, attr) + key); return

    # validation / commit

    def _commit(self):
        ok = True

        try:
            v = int(self._r_buf)
            if v < 2: raise ValueError
            self.r = v
        except ValueError:
            self._error_msg   = f"r must be integer >= 2  (got '{self._r_buf}')"
            self._error_timer = 120
            self._r_buf = str(self.r); ok = False

        try:
            v = int(self._n_buf)
            if v < 1: raise ValueError
            self.n = v
        except ValueError:
            if ok:
                self._error_msg   = f"n must be integer >= 1  (got '{self._n_buf}')"
                self._error_timer = 120
            self._n_buf = str(self.n); ok = False

        try:
            v = int(self._np_buf)
            if v < 1: raise ValueError
            self.num_particles = v
        except ValueError:
            if ok:
                self._error_msg   = "particles must be integer >= 1"
                self._error_timer = 120
            self._np_buf = str(self.num_particles); ok = False

        for attr, buf_attr, label, lo in [
            ('w',  '_w_buf',  'w',  None),
            ('c1', '_c1_buf', 'c1', 0.0),
            ('c2', '_c2_buf', 'c2', 0.0),
        ]:
            try:
                v = float(getattr(self, buf_attr))
                if lo is not None and v < lo: raise ValueError
                setattr(self, attr, v)
            except ValueError:
                if ok:
                    constraint = f">= {lo}" if lo is not None else "a number"
                    self._error_msg   = f"{label} must be float {constraint}"
                    self._error_timer = 120
                setattr(self, buf_attr, f"{getattr(self, attr):.4f}"); ok = False

        if self.show_droste_params:
            try:
                v = float(self._q_buf)
                if v <= 0: raise ValueError
                self.q = v
            except ValueError:
                if ok:
                    self._error_msg   = f"q must be float > 0  (got '{self._q_buf}')"
                    self._error_timer = 120
                self._q_buf = f"{self.q:.6f}"; ok = False

            try:
                v = int(self._bit_shift_buf)
                if v < 1: raise ValueError
                self.bit_shift = v
            except ValueError:
                if ok:
                    self._error_msg   = "bit_shift must be integer >= 1"
                    self._error_timer = 120
                self._bit_shift_buf = str(self.bit_shift); ok = False

        if ok:
            self._apply_params()
        return ok

    # standalone run

    def run(self):
        from DynamicOptimisers import PSO
        if self._problem_class is DynamicOneMax:
            self._sa_prob = self._problem_class(r=self.r, n=self.n, q=self.q, bit_shift=self.bit_shift)
        else:
            self._sa_prob = self._problem_class(r=self.r, n=self.n, rate=self.rate)
        self._sa_opt  = PSO(self.r, self.n, self.num_particles)
        self._sa_gen  = self._sa_prob.run_problem()

        def _draw():
            ps = next(self._sa_gen)
            (gbv, gbp, pbv, pbp, pos, vel, pf, expl) = \
                self._sa_opt.iterate_candidate(ps['optimum'])
            self.gbest_history.append(gbv)
            if self.n in (2, 3):
                for i in range(self.num_particles):
                    self.trail_history[i].append(pos[i].copy())
            self.draw_frame(gbv, gbp, pbv, pbp, pos, vel, pf, expl,
                            ps['optimum'], ps['iteration'],
                            ps['optimum_moved'], is_running=True)
            time.sleep(self.wait_time)

        py5.run_sketch(block=False, sketch_functions={
            'settings': self.settings, 'setup': self.setup, 'draw': _draw,
        })