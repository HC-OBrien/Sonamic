"""
Combined audiovisual displays.

    • A single opt_alg and problem_generator drives everything.
    • Audio is fired from draw_and_sound() calling pathway classes from
      Sonifications.py directly, sent to the generic AudioState.
    • The visuals.on_params_changed callback is wired so that when the
      user edits parameters in the ParamEditor, those changes propagate
      back to this class's opt_alg and problem_generator.
    • show_droste_params is derived by checking if the problem class is
      DynamicOneMax — it is not passed as a separate constructor argument.
"""

import py5
import numpy as np

from DynamicProblems import DynamicOneMax
from Visualisations import VisualiseOPO, VisualisePSO
from Synthesizer import AudioEngine
from Sonifications import (OneShotSoundbank, HallReverb,
                           FitnessPathway, PositionPathway, MutationPathway)

try:
    from pythonosc import udp_client
    _HAS_OSC = True
except ImportError:
    _HAS_OSC = False


#  AudioVisualiseOPO — combined audio + visual for the (1+1) EA

class AudioVisualiseOPO:
    """
    Combines VisualiseOPO with audio (Py synth or OSC).
    One opt_alg and one problem_generator are authoritative.
    Pathways and one-shot sounds are wired directly from Sonifications classes.
    """

    def __init__(self, problem, opt_alg,
                 sound_engine='Py', audio_engine=None,
                 r=2, n=10, p=None, q=0.023, bit_shift=1,
                 rate=200, wait_time=0.1, pathway='mutation'):

        # parameters
        self.r             = r
        self.n             = n
        self.rate          = rate
        self.p             = p
        self.q             = q
        self.bit_shift     = bit_shift
        self.wait_time     = wait_time
        self.sound_engine  = sound_engine
        self.pathway       = pathway

        # class references
        self._problem_class = problem
        self._opt_alg_class = opt_alg

        # single authoritative algorithm and problem
        self.opt_alg = opt_alg(r=self.r, n=self.n, p=self.p)
        if problem is DynamicOneMax:
            self.problem = problem(r=self.r, n=self.n, q=self.q, bit_shift=self.bit_shift)
        else:
            self.problem = problem(r=self.r, n=self.n, rate=self.rate)
        self.problem_generator = self.problem.run_problem()

        # visualiser
        self.visuals = VisualiseOPO(
            problem, opt_alg,
            n=n, r=r, p=p, q=q, bit_shift=bit_shift,
            rate=rate, wait_time=wait_time,
        )

        # param change callback
        self.visuals.on_params_changed = self._on_params_changed

        # audio engine
        if audio_engine is not None:
            self.engine = audio_engine
        else:
            self.engine = AudioEngine()
            self.engine.start()

        # hall reverb
        self._reverb = HallReverb(self.engine.state.sample_rate)

        # audio pathways
        self._oneshots        = None
        self.mutation_pathway = False
        self.position_pathway = False
        self.fitness_pathway  = False
        self._setup_audio()

        # cache for freeze-frame
        self._last_candidate = None
        self._last_optimum   = None
        self._last_iteration = 0
        self._last_fitness   = 0

        # recording
        self.recorder = None

    # audio setup

    def _setup_audio(self):
        """Initialise pathways for the current sound_engine and pathway selection."""
        if self.sound_engine == 'OSC':
            if not _HAS_OSC:
                raise ImportError("python-osc is not installed.")
            self.client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
        elif self.sound_engine == 'Py':
            self._reverb.wrap(self.engine.state)
            self._oneshots = OneShotSoundbank(self.engine.state.sample_rate)
            _pw = self.pathway.lower()
            if _pw == 'mutation':
                self.mutation_pathway = MutationPathway(self.n, self.r, self.engine)
            elif _pw == 'position':
                self.position_pathway = PositionPathway(self.n, self.r, self.engine)
            elif _pw == 'fitness':
                self.fitness_pathway  = FitnessPathway(audio_engine=self.engine)

    def _stop_audio(self):
        """Stop all active pathway threads, reset pathway flags, unwrap reverb."""
        if self.mutation_pathway:
            self.mutation_pathway.stop()
        if self.position_pathway:
            self.position_pathway.stop()
        if self.fitness_pathway:
            self.fitness_pathway.stop()
        self.mutation_pathway = False
        self.position_pathway = False
        self.fitness_pathway  = False
        self._reverb.unwrap(self.engine.state)

    # param change callback

    def _on_params_changed(self, new_r, new_n, new_p, new_q, new_bit_shift):
        """
        Called when the user changes parameters in the editor.
        Re-creates our opt_alg and problem to match the new parameters.
        """
        self.r         = new_r
        self.n         = new_n
        self.p         = new_p
        self.q         = new_q
        self.bit_shift = new_bit_shift

        # Rebuild algorithm and problem with new params
        self.opt_alg = self._opt_alg_class(r=self.r, n=self.n, p=self.p)
        if self._problem_class is DynamicOneMax:
            self.problem = self._problem_class(r=self.r, n=self.n, q=self.q, bit_shift=self.bit_shift)
        else:
            self.problem = self._problem_class(r=self.r, n=self.n, rate=self.rate)
        self.problem_generator = self.problem.run_problem()

        # Rebuild audio pathways with new params
        self._stop_audio()
        self._setup_audio()

        # Reset cache (recorder is intentionally kept — mid-run edits don't
        # interrupt an in-progress recording)
        self._last_candidate = None
        self._last_optimum   = None
        self._last_iteration = 0
        self._last_fitness   = 0

        print(f"[AudioVisualiseOPO] Params updated: r={self.r}, n={self.n}, "
              f"p={self.p}, q={self.q}, bit_shift={self.bit_shift}")

    # audio sending

    def send_sound(self, current_candidate, current_fitness, old_candidate,
                   old_fitness, mutant_taken, fitness_array, optimum,
                   iteration, optimum_moved):
        if self.sound_engine == 'Py':
            self._sound_py(current_candidate, current_fitness, old_candidate,
                           old_fitness, mutant_taken, fitness_array, optimum,
                           iteration, optimum_moved)
        elif self.sound_engine == 'OSC':
            self._sound_osc(current_candidate, current_fitness, old_candidate,
                            old_fitness, mutant_taken, fitness_array, optimum,
                            iteration, optimum_moved)

    def _sound_py(self, current_candidate, current_fitness, old_candidate,
                  old_fitness, mutant_taken, fitness_array, optimum,
                  iteration, optimum_moved):
        if mutant_taken and self.mutation_pathway:
            self.mutation_pathway.on_mutation(
                current_candidate, current_fitness,
                old_candidate,     old_fitness,
            )

        if self.position_pathway:
            self.position_pathway.update(current_candidate)

        if self.fitness_pathway:
            max_fitness = max(self.n * (self.r // 2), 1)
            self.fitness_pathway.update(current_fitness, max_fitness)

        if current_fitness == 0 and old_fitness != 0:
            self.engine.state.queue_oneshot(self._oneshots.optimum_ding())

        if optimum_moved and not isinstance(self.problem, DynamicOneMax):
            self.engine.state.queue_oneshot(self._oneshots.optimum_moved_gong())

    def _sound_osc(self, current_candidate, current_fitness, old_candidate,
                   old_fitness, mutant_taken, fitness_array, optimum,
                   iteration, optimum_moved):
        if optimum_moved:
            self.client.send_message("/optimum_moved", 1)
        if current_fitness != old_fitness:
            fitness_param = np.minimum(100, current_fitness * (100 / max(self.n, 1)))
            self.client.send_message("/fitness", float(fitness_param))
        if iteration % 50 == 0:
            for i in range(self.n):
                self.client.send_message(f"/lfo{i}", int(current_candidate[i]))
        if iteration % 50 == 0:
            for i in range(self.n):
                self.client.send_message(f"/vol{i}", float(fitness_array[i] / max(self.r, 1)))
        if mutant_taken:
            self.client.send_message("/mutant_taken", 1)
        if current_fitness == 0 and current_fitness != old_fitness:
            self.client.send_message("/optimum_found", 1)

    # main draw+sound loop

    def draw_and_sound(self):
        """Advance one step, draw the frame, fire audio."""
        problem_state = next(self.problem_generator)
        optimum       = problem_state['optimum']
        optimum_moved = problem_state['optimum_moved']
        iteration     = problem_state['iteration']

        (current_candidate, current_fitness, old_candidate, old_fitness,
         mutant_taken, fitness_array) = self.opt_alg.iterate_candidate(optimum)

        # Record fitness for the graph
        self.visuals.distance_history.append(current_fitness)

        # Cache for freeze-frame
        self._last_candidate = current_candidate
        self._last_optimum   = optimum
        self._last_iteration = iteration
        self._last_fitness   = current_fitness

        # Draw
        self.visuals.draw_frame(current_candidate, optimum, iteration,
                                current_fitness, is_running=True)

        # Sound
        self.send_sound(current_candidate, current_fitness, old_candidate,
                        old_fitness, mutant_taken, fitness_array,
                        optimum, iteration, optimum_moved)

        # recording
        if self.recorder is not None:
            max_fit = self.r * self.n
            synth_freq = None
            first_ding = None
            if mutant_taken:
                frac = 1.0 - min(current_fitness / max(max_fit, 1), 1.0)
                synth_freq = 220.0 + frac * (880.0 - 220.0)
            ding_fired = (current_fitness == 0 and old_fitness != 0)
            if ding_fired:
                first_ding = iteration
            gong_fired = (optimum_moved and not self.visuals.show_droste_params)
            self.recorder.record_opo_step(
                iteration, current_fitness, old_fitness,
                mutant_taken, optimum_moved,
                synth_freq=synth_freq,
                ding_fired=ding_fired,
                gong_fired=gong_fired,
                first_ding=first_ding,
            )

    def draw_frame_only(self):
        """Redraw last known frame (paused) or idle placeholder."""
        if self._last_candidate is None:
            self.visuals.draw_idle()
        else:
            self.visuals.draw_frame(
                self._last_candidate, self._last_optimum,
                self._last_iteration, self._last_fitness,
                is_running=False,
            )

    # input forwarding

    def mouse_pressed(self, mx, my):
        self.visuals.mouse_pressed(mx, my, is_running=False)

    def key_pressed(self, key, key_code):
        self.visuals.key_pressed(key, key_code, is_running=False)

    # standalone run

    def run(self):
        py5.run_sketch(block=True, sketch_functions={
            'settings': self.visuals.settings,
            'setup':    self.visuals.setup,
            'draw':     self.draw_and_sound,
        })


#  AudioVisualisePSO — combined audio + visual for PSO

class AudioVisualisePSO:
    """
    Combines VisualisePSO with audio (Py synth or OSC).
    Same design — one opt_alg, one problem_generator, param callback wired.
    Pathways and one-shot sounds are wired directly from Sonifications classes.
    """

    def __init__(self, problem, opt_alg,
                 sound_engine='Py', audio_engine=None,
                 num_particles=8, n=3, r=100, rate=100, wait_time=0.1,
                 w=0.5, c1=1.8, c2=1.8, q=0.1, bit_shift=10,
                 pathway='position'):

        # store parameters
        self.num_particles = num_particles
        self.n             = n
        self.r             = r
        self.rate          = rate
        self.wait_time     = wait_time
        self.sound_engine  = sound_engine
        self.w             = w
        self.c1            = c1
        self.c2            = c2
        self.q             = q
        self.bit_shift     = bit_shift
        self.pathway       = pathway

        # class references
        self._problem_class = problem
        self._opt_alg_class = opt_alg

        # visualiser
        self.visuals = VisualisePSO(
            problem, opt_alg,
            num_particles=num_particles, n=n, r=r,
            w=w, c1=c1, c2=c2, q=q, bit_shift=bit_shift,
            rate=rate, wait_time=wait_time,
        )

        # algorithm and problem
        self.opt_alg = opt_alg(self.r, self.n, self.num_particles,
                               w=self.w, c1=self.c1, c2=self.c2)
        if problem is DynamicOneMax:
            self.problem = problem(r=self.r, n=self.n, q=self.q, bit_shift=self.bit_shift)
        else:
            self.problem = problem(r=self.r, n=self.n, rate=self.rate)
        self.problem_generator = self.problem.run_problem()

        # param change callback
        self.visuals.on_params_changed = self._on_params_changed

        # audio engine
        if audio_engine is not None:
            self.engine = audio_engine
        else:
            self.engine = AudioEngine()
            self.engine.start()

        # hall reverb insert
        self._reverb = HallReverb(self.engine.state.sample_rate)

        # audio pathways
        self._oneshots         = None
        self.position_pathway  = False
        self.mutation_pathway  = False
        self.fitness_pathway   = False
        self._prev_gbest_value = None
        self._setup_audio()

        # cache for freeze-frame
        self._last_gbest_value     = 0.0
        self._last_gbest_position  = None
        self._last_pbest_values    = None
        self._last_pbest_positions = None
        self._last_positions       = None
        self._last_velocities      = None
        self._last_position_fits   = None
        self._last_optimum         = None
        self._last_iteration       = 0
        self._last_optimum_moved   = False

        # recording
        self.recorder = None

    # audio setup

    def _setup_audio(self):
        """Initialise pathways for the current sound_engine and pathway selection."""
        if self.sound_engine == 'OSC':
            if not _HAS_OSC:
                raise ImportError("python-osc is not installed.")
            self.client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
        elif self.sound_engine == 'Py':
            self._reverb.wrap(self.engine.state)
            self._oneshots = OneShotSoundbank(self.engine.state.sample_rate)
            _pw = self.pathway.lower()
            if _pw == 'position':
                self.position_pathway = PositionPathway(self.n, self.r, self.engine)
            elif _pw == 'fitness':
                self.fitness_pathway  = FitnessPathway(audio_engine=self.engine)
            elif _pw == 'mutation':
                pass  # PSO has no mutations; only the always-on one-shots will play

    def _stop_audio(self):
        """Stop all active pathway threads, reset pathway flags, unwrap reverb."""
        if self.position_pathway:
            self.position_pathway.stop()
        if self.fitness_pathway:
            self.fitness_pathway.stop()
        if self.mutation_pathway:
            self.mutation_pathway.stop()
        self.position_pathway = False
        self.mutation_pathway = False
        self.fitness_pathway  = False
        self._reverb.unwrap(self.engine.state)

    # param change callback

    def _on_params_changed(self, new_r, new_n, new_num_particles,
                           new_w, new_c1, new_c2, new_q, new_bit_shift):
        self.r             = new_r
        self.n             = new_n
        self.num_particles = new_num_particles
        self.w             = new_w
        self.c1            = new_c1
        self.c2            = new_c2
        self.q             = new_q
        self.bit_shift     = new_bit_shift

        # Rebuild algorithm and problem
        self.opt_alg = self._opt_alg_class(
            r=self.r, n=self.n, num_particles=self.num_particles,
            w=self.w, c1=self.c1, c2=self.c2)
        if self._problem_class is DynamicOneMax:
            self.problem = self._problem_class(r=self.r, n=self.n, q=self.q, bit_shift=self.bit_shift)
        else:
            self.problem = self._problem_class(r=self.r, n=self.n, rate=self.rate)
        self.problem_generator = self.problem.run_problem()

        # Rebuild audio pathways with new params
        self._stop_audio()
        self._setup_audio()

        # Reset cache
        self._last_positions = None

        print(f"[AudioVisualisePSO] Params updated: r={self.r}, n={self.n}, "
              f"np={self.num_particles}, w={self.w}, q={self.q}, bit_shift={self.bit_shift}")

    # audio sending

    def send_sound(self, gbest_value, gbest_position, pbest_values,
                   pbest_positions, positions, velocities, position_fits,
                   optimum, iteration, optimum_moved):
        if self.sound_engine == 'Py':
            self._sound_py(gbest_value, gbest_position, pbest_values,
                           pbest_positions, positions, velocities, position_fits,
                           optimum, iteration, optimum_moved)
        elif self.sound_engine == 'OSC':
            self._sound_osc(gbest_value, gbest_position, pbest_values,
                            pbest_positions, positions, velocities, position_fits,
                            optimum, iteration, optimum_moved)

    def _sound_py(self, gbest_value, gbest_position, pbest_values,
                  pbest_positions, positions, velocities, position_fits,
                  optimum, iteration, optimum_moved):
        if self.position_pathway:
            raw_diversity = float(np.std(np.asarray(positions, dtype=np.float64),
                                         axis=0).mean())
            max_std = max((self.r - 1) / 2.0, 1.0)
            diversity_norm = float(np.clip(raw_diversity / max_std, 0.0, 1.0))
            self.position_pathway.update(gbest_position, diversity_norm)

        if self.fitness_pathway:
            max_fitness = max(self.n * (self.r // 2), 1)
            self.fitness_pathway.update(gbest_value, max_fitness)

        prev = self._prev_gbest_value
        if gbest_value == 0 and (prev is None or prev != 0):
            self.engine.state.queue_oneshot(self._oneshots.optimum_ding())
        self._prev_gbest_value = gbest_value

        if optimum_moved and not isinstance(self.problem, DynamicOneMax):
            self.engine.state.queue_oneshot(
                self._oneshots.optimum_moved_gong(base_freq=55.0, amplitude=0.6)
            )

    def _sound_osc(self, gbest_value, gbest_position, pbest_values,
                   pbest_positions, positions, velocities, position_fits,
                   optimum, iteration, optimum_moved):
        if optimum_moved:
            self.client.send_message("/optimum_moved", 1)
        if iteration % 1000 == 0:
            self.client.send_message("/global_best", float(gbest_value))
            for i in range(self.num_particles):
                self.client.send_message(f"/particle_best_fitness{i}", float(pbest_values[i]))
                self.client.send_message(f"/speed{i}",
                    float(100 * abs(float(np.mean(velocities[i])))))
                self.client.send_message(f"/note{i}", float(position_fits[i]))

    # main draw+sound loop

    def draw_and_sound(self):
        """Advance PSO one step, draw, fire audio."""
        problem_state = next(self.problem_generator)
        optimum       = problem_state['optimum']
        optimum_moved = problem_state['optimum_moved']
        iteration     = problem_state['iteration']

        (gbest_value, gbest_position, pbest_values, pbest_positions,
         positions, velocities, position_fits) = self.opt_alg.iterate_candidate(optimum)

        # Record for graph and trails
        self.visuals.gbest_history.append(gbest_value)
        if self.n in (2, 3):
            for i in range(self.num_particles):
                self.visuals.trail_history[i].append(positions[i].copy())

        # Cache
        self._last_gbest_value     = gbest_value
        self._last_gbest_position  = gbest_position
        self._last_pbest_values    = pbest_values
        self._last_pbest_positions = pbest_positions
        self._last_positions       = positions
        self._last_velocities      = velocities
        self._last_position_fits   = position_fits
        self._last_optimum         = optimum
        self._last_iteration       = iteration
        self._last_optimum_moved   = optimum_moved

        # Draw
        self.visuals.draw_frame(
            gbest_value, gbest_position, pbest_values, pbest_positions,
            positions, velocities, position_fits,
            optimum, iteration, optimum_moved, is_running=True,
        )

        # Sound
        self.send_sound(
            gbest_value, gbest_position, pbest_values, pbest_positions,
            positions, velocities, position_fits,
            optimum, iteration, optimum_moved,
        )

        # recording
        if self.recorder is not None:
            gong_fired = (optimum_moved and not self.visuals.show_droste_params)
            if self.position_pathway:
                tone_freqs, tone_amps = \
                    self.position_pathway._compute_tones(gbest_position)
            else:
                tone_freqs = np.array([])
                tone_amps  = np.array([])
            self.recorder.record_pso_step(
                iteration, gbest_value, position_fits,
                positions, optimum_moved,
                gong_fired=gong_fired,
                tone_freqs=tone_freqs,
                tone_amps=tone_amps,
            )

    def draw_frame_only(self):
        """Redraw last known frame or idle placeholder."""
        if self._last_positions is None:
            self.visuals.draw_idle()
        else:
            self.visuals.draw_frame(
                self._last_gbest_value, self._last_gbest_position,
                self._last_pbest_values, self._last_pbest_positions,
                self._last_positions, self._last_velocities,
                self._last_position_fits,
                self._last_optimum, self._last_iteration,
                self._last_optimum_moved, is_running=False,
            )

    # input forwarding

    def mouse_pressed(self, mx, my):
        self.visuals.mouse_pressed(mx, my, is_running=False)

    def key_pressed(self, key, key_code):
        self.visuals.key_pressed(key, key_code, is_running=False)

    # standalone run

    def run(self):
        py5.run_sketch(block=True, sketch_functions={
            'settings': self.visuals.settings,
            'setup':    self.visuals.setup,
            'draw':     self.draw_and_sound,
        })
