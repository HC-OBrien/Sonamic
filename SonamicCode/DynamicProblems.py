"""
Dynamic test problem generator file.

Three movement strategies are provided:
    • DynamicOneMax — each dimension of the optimum independently mutates
                        by ±bit_shift with probability q per iteration,
                        producing a slow continuous drift.
    • RandomIntStr  — every `rate` iterations the optimum jumps to a
                        uniformly random string.
    • AllToNone  — the optimum alternates between the all-(r-1) string
                        and the all-zero string every `rate` iterations.
"""

import numpy as np

class DynamicOneMax:

    def __init__(self, r=2, n=10, q=None, bit_shift=1):
        """
        :param r: int >= 2, max dimension/variable size
        :param n: int, number of dimensions/variables
        :param q: float in [0,1], probability of an element being shifted
        :param bit_shift: int, fixed amount that elements are shifted by
        """

        # problem dimensions
        self.r = r              # alphabet size (values per dimension)
        self.n = n              # string length  (number of dimensions)
        self.q             = q if q is not None else np.log(n) / (n**2)
        self.bit_shift     = bit_shift

        # initial problem
        self.current_optimum = np.random.randint(0, self.r, size=self.n)
        self.optimum_moved = True
        self.iteration  = 0        # current iteration counter

        # run state
        self.is_running = True     # set False to stop the generator

    def iterate_optimum(self):

        # Decide which dimensions mutate this iteration
        change_mask = np.random.rand(self.n) < self.q

        # Random direction for each dimension: +bit_shift or -bit_shift
        directions = np.random.choice(
            [-self.bit_shift, self.bit_shift], size=self.n
        )

        # Apply mutations with modular wrap-around
        new_optimum = (self.current_optimum + change_mask * directions) % self.r

        return new_optimum

    def run_problem(self):
        """
        dict with keys:
            'optimum'       : np.ndarray — current target string
            'iteration'     : int        — iteration counter
            'optimum_moved' : bool       — True if optimum changed this step
        """
        try:
            while self.is_running:
                # Move the optimum according to the chosen strategy
                self.current_optimum = self.iterate_optimum()

                # Yield the current state to the caller
                yield {
                    'optimum':       self.current_optimum,
                    'iteration':     self.iteration,
                    'optimum_moved': self.optimum_moved,
                }

                self.iteration += 1

        except KeyboardInterrupt:
            self.is_running = False
        finally:
            print("TEST PROBLEM STOPPED")


class RandomIntStr:

    def __init__(self, r=2, n=10, rate=200):
        """
        :param r: int >= 2, max dimension/variable size
        :param n: int, number of dimensions/variables
        :param rate: rate at which the optimum changes
        """

        # problem dimensions
        self.r = r  # alphabet size (values per dimension)
        self.n = n  # string length  (number of dimensions)
        self.rate = rate

        # initialise problem
        self.optimum_moved = False
        self.current_optimum = np.random.randint(0, self.r, size=self.n)
        self.iteration  = 0        # current iteration counter

        # run state
        self.is_running = True     # set False to stop the generator

    def iterate_optimum(self):
        """
        Every `rate` iterations, jump to a completely new random optimum.
        Between jumps the optimum stays fixed.
        """
        self.optimum_moved = False
        new_optimum = self.current_optimum.copy()

        if self.iteration % self.rate == 0:
            self.optimum_moved = True
            new_optimum = np.random.randint(0, self.r, size=self.n)

        return new_optimum

    def run_problem(self):
        """
        dict with keys:
            'optimum'       : np.ndarray — current target string
            'iteration'     : int        — iteration counter
            'optimum_moved' : bool       — True if optimum changed this step
        """
        try:
            while self.is_running:

                # Move the optimum according to the chosen strategy
                self.current_optimum = self.iterate_optimum()

                # Yield the current state to the caller
                yield {
                    'optimum':       self.current_optimum,
                    'iteration':     self.iteration,
                    'optimum_moved': self.optimum_moved,
                }

                self.iteration += 1

        except KeyboardInterrupt:
            self.is_running = False
        finally:
            print("TEST PROBLEM STOPPED")


class AllToNone:

    def __init__(self, r=2, n=10, rate=200):
        """
        :param r: int >= 2, max dimension/variable size
        :param n: int, number of dimensions/variables
        :param rate: rate at which the optimum changes
        """

        # problem dimensions
        self.r = r  # alphabet size (values per dimension)
        self.n = n  # string length  (number of dimensions)
        self.rate = rate

        # initialise problem
        self.optimum_moved = False
        self.current_optimum = np.array([self.r - 1] * self.n)
        self.iteration  = 0        # current iteration counter

        # run state
        self.is_running = True     # set False to stop the generator

    def iterate_optimum(self):

        self.optimum_moved = False
        new_optimum = self.current_optimum.copy()

        if self.iteration % self.rate == 0:
            self.optimum_moved = True
            # If currently all-zero, switch to all-(r-1) and vice versa
            if np.array_equal(self.current_optimum, np.zeros(self.n)):
                new_optimum = np.array([self.r - 1] * self.n)
            else:
                new_optimum = np.zeros(self.n)

        return new_optimum

    def run_problem(self):

        try:
            while self.is_running:
                # Move the optimum according to the chosen strategy
                self.current_optimum = self.iterate_optimum()

                # Yield the current state to the caller
                yield {
                    'optimum':       self.current_optimum,
                    'iteration':     self.iteration,
                    'optimum_moved': self.optimum_moved,
                }

                self.iteration += 1

        except KeyboardInterrupt:
            self.is_running = False
        finally:
            print("TEST PROBLEM STOPPED")
