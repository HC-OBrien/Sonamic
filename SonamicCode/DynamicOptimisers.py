"""
Optimisation algorithms for dynamic problems.

Two algorithms are provided:

OPO — (1+1) Evolutionary Algorithm
    A minimal evolutionary strategy that maintains a single candidate solution.
    Each iteration it mutates the candidate and keeps the mutant if it is at
    least as good (≤ fitness) as the parent.  Fitness is the summed per-dimension
    distance to the optimum on a circular alphabet.

PSO — Particle Swarm Optimisation
    A population of particles fly through a continuous [0, r]^n search space.
    Each particle is attracted toward its personal best position and the global
    best position found by the swarm.  An exploration term scaled by the current
    global-best fitness adds noise to encourage escape from local optima.
"""

import numpy as np

#  OPO — (1+1) Evolutionary Algorithm

class OPO:
    """
    (1+1) Evolutionary Algorithm for the generalised OneMax problem.

    Maintains a single candidate string of length n over alphabet {0,…,r-1}.
    Mutates using same stochastic process as DynamicOneMax
    Fitness is measured as the total circular distance to a given optimum.
    """

    def __init__(self, r=2, n=10, p=None):
        """
        :param r: int >= 2, max dimension/variable size
        :param n: int, number of dimensions/variables
        :param p: float, mutation probability
        """

        self.r = r  # alphabet size
        self.n = n   # string length
        self.p = p if p is not None else 1 / n   # Mutation rate

        # Initialise candidate uniformly at random over the alphabet
        self.current_candidate = np.random.randint(0, self.r, size=self.n)

        # These are set after the first iteration
        self.current_fitness = None    # scalar total distance
        self.fitness_array   = None    # per-dimension distances

    # fitness evaluation

    def fitness_check(self, candidate, optimum):

        # Forward wrap: how far from candidate to optimum going "up"
        forward  = (optimum - candidate) % self.r

        # Backward wrap: how far going "down"
        backward = (candidate - optimum) % self.r

        # Take the shorter of the two directions per dimension
        distances = np.minimum(forward, backward)

        return int(np.sum(distances)), distances.tolist()

    # mutation operator

    def mutate(self, candidate):

        mutated = candidate.copy()

        # Draw one uniform random number per dimension
        rng = np.random.random(self.n)

        # Lower half of mutation probability → increment by 1
        less = rng < self.p * 0.5
        mutated[less] = (mutated[less] + 1) % self.r

        # Upper half of mutation probability → decrement by 1
        more = (rng >= self.p * 0.5) & (rng < self.p)
        mutated[more] = (mutated[more] - 1) % self.r

        return mutated

    # one iteration of the (1+1) EA

    def iterate_candidate(self, optimum):
        """
        Returns tuple of:
            current_candidate : np.ndarray — the (possibly updated) solution
            current_fitness   : int        — total distance to optimum
            old_candidate     : np.ndarray — the parent before this iteration
            old_fitness       : int        — parent's fitness
            mutant_taken      : bool       — True if the mutant replaced parent
            fitness_array     : list       — per-dimension distances
        """
        # Snapshot the current (parent) solution
        old_candidate = self.current_candidate.copy()
        old_fitness, old_fitness_array = self.fitness_check(
            self.current_candidate, optimum
        )

        # Create a mutant offspring
        mutant = self.mutate(self.current_candidate)
        mutant_fitness, mutant_fitness_array = self.fitness_check(
            mutant, optimum
        )

        # Selection: accept mutant if it is at least as good (< distance)
        if mutant_fitness < old_fitness:
            self.current_candidate = mutant
            self.current_fitness   = mutant_fitness
            self.fitness_array     = mutant_fitness_array
            mutant_taken           = True
        else:
            # Keep the parent unchanged
            self.current_candidate = old_candidate
            self.current_fitness   = old_fitness
            self.fitness_array     = old_fitness_array
            mutant_taken           = False

        return (
            self.current_candidate,
            self.current_fitness,
            old_candidate,
            old_fitness,
            mutant_taken,
            self.fitness_array,
        )


#  PSO — Particle Swarm Optimisation

class PSO:
    """
    A swarm of particles moves through [0, r]^n continuous space.
    Each particle's velocity is updated based on four components:
        • Inertia    — momentum from the previous velocity
        • Cognitive  — attraction toward the particle's personal best
        • Social     — attraction toward the global best of the swarm
        • Charge     — pairwise repulsion from nearby charged particles
                       (Blackwell & Bentley, 2002 — collision avoidance)
    """

    def __init__(self, r=2, n=10, num_particles=8,
                 w=0.2, c1=0.5, c2=0.5,
                 charges=None, p_core=1e-6, p=None):
        """
        :param r: int >= 2, max dimension/variable size
        :param n: int, number of dimensions/variables
        :param num_particles: number of swarm particles
        :param w: float, inertia weight
        :param c1: float, cognitive acceleration coefficient
        :param c2: float, social acceleration coefficient
        :param charges: array, index of charge identities
        :param p_core: float, lower cut-off distance
        :param p: float, upper cut-off distance
        """

        self.n             = n
        self.r             = r
        self.num_particles = num_particles
        self.w  = w
        self.c1 = c1
        self.c2 = c2

        # Charge parameters
        if charges is None:
            self.charges = np.ones(num_particles)   # all particles fully charged
        else:
            self.charges = np.asarray(charges, dtype=np.float64)
        self.p_core = p_core
        self.p      = float(r) if p is None else float(p)

        # initialise particles
        # positions:
        self.positions  = np.random.uniform(0, r, (num_particles, n))
        # Velocities:
        self.velocities = np.random.uniform(-r / 2, r / 2, (num_particles, n))
        # Current fitness for each particle (inf = not yet evaluated)
        self.position_fits = np.full(num_particles, np.inf)

        # personal bests
        self.pbest_positions = self.positions.copy()
        self.pbest_values    = np.full(num_particles, np.inf)

        # global best
        self.gbest_position = self.positions[0].copy()
        self.gbest_value    = np.inf

        # sentinel particle
        self.sentinel = np.random.randint(0, self.r, size=self.n)
        self.sentinel_fitness = np.inf

    # evaluate all particles against the optimum

    def fitness_check(self, candidate, optimum):
        fitness = float(np.linalg.norm(optimum - candidate))
        return fitness

    def evaluate_particles(self, optimum):
        """
        Compute fitness for every particle and update personal / global bests.

        Fitness = sum of absolute differences to the optimum (L1 distance).
        Lower is better.
        """
        for i in range(self.num_particles):
            fitness = self.fitness_check(self.positions[i], optimum)
            self.position_fits[i] = fitness

            if fitness < self.pbest_values[i]:
                self.pbest_values[i]    = fitness
                self.pbest_positions[i] = self.positions[i].copy()

            if fitness < self.gbest_value:
                self.gbest_value    = fitness
                self.gbest_position = self.positions[i].copy()

    # boundary enforcement

    def enforce_box(self, position, velocity):

        hit_low  = (position <= 1e-6) & (velocity < 0)
        hit_high = (position >= self.r - 1e-6) & (velocity > 0)

        velocity[hit_low]  = -velocity[hit_low]  * 0.9
        velocity[hit_high] = -velocity[hit_high] * 0.9

        return velocity

    # charge acceleration (collision avoidance)

    def charge_acceleration(self, i):
        """
        Compute the repulsive charge acceleration on particle i.
        The sum is restricted to pairs satisfying  p_core < r_ij < p.
        Pairs outside this shell contribute nothing.
        """
        Q_i = self.charges[i]
        if Q_i == 0.0:
            return np.zeros(self.n)

        acc = np.zeros(self.n)
        for j in range(self.num_particles):
            if j == i:
                continue
            Q_j = self.charges[j]
            if Q_j == 0.0:
                continue

            r_vec = self.positions[i] - self.positions[j]   # displacement vector
            r_dist = np.linalg.norm(r_vec)                  # scalar distance

            # Only apply repulsion within the shell (p_core, p)
            if self.p_core < r_dist < self.p:
                acc += (Q_i * Q_j / r_dist ** 3) * r_vec

        return acc

    # one iteration of PSO

    def iterate_candidate(self, optimum):
        """
        Returns tuple of:
            gbest_value     : float
            gbest_position  : np.ndarray
            pbest_values    : np.ndarray (copy)
            pbest_positions : np.ndarray (copy)
            positions       : np.ndarray (copy)
            velocities      : np.ndarray (copy)
            position_fits   : np.ndarray (copy)
        """
        old_sentinel_fitness = self.sentinel_fitness
        self.sentinel_fitness = self.fitness_check(self.sentinel, optimum)
        if self.sentinel_fitness != old_sentinel_fitness:
            for i in range(self.num_particles):
                self.pbest_values[i] = float(
                    np.linalg.norm(optimum - self.positions[i])
                )
                self.pbest_positions[i] = self.positions[i].copy()

            best = int(np.argmin(self.pbest_values))
            self.gbest_position = self.pbest_positions[best].copy()
            self.gbest_value    = float(self.pbest_values[best])
        else:
            self.evaluate_particles(optimum)

        # update velocities and positions
        for i in range(self.num_particles):

            # Inertia component
            inertia   = self.w * self.velocities[i]

            # Cognitive component — pull toward personal best
            cognitive = np.random.rand(self.n) * self.c1 * (self.pbest_positions[i] - self.positions[i])

            # Social component — pull toward global best
            social    = np.random.rand(self.n) * self.c2 * (self.gbest_position - self.positions[i])

            # Charge component — repulsion from nearby charged particles
            charge    = self.charge_acceleration(i)

            # Velocity update: inertia + cognitive + social + charge
            self.velocities[i] = inertia + cognitive + social + charge

            # Reflect velocity at boundaries
            self.velocities[i] = self.enforce_box(
                self.positions[i], self.velocities[i]
            )

            # Position update
            self.positions[i] = self.positions[i] + self.velocities[i]

        return (
            self.gbest_value,
            self.gbest_position,
            self.pbest_values.copy(),
            self.pbest_positions.copy(),
            self.positions.copy(),
            self.velocities.copy(),
            self.position_fits.copy(),
        )
