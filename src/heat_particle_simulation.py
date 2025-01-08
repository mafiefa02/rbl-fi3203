import jax.numpy as jnp
from jax import random, vmap
import numpy as np


class HeatParticleSimulation:
    def __init__(self, grid_size, num_particles, seed=0):
        self.m = grid_size
        self.num_particles = num_particles
        self.master_key = random.key(seed)

        # Initialize grid and particles
        self.grid = jnp.zeros((self.m, self.m))

        # Set initial position to center of grid
        px = self.m // 2
        py = self.m // 2

        # Set initial positions and opacities
        self.grid = self.grid.at[py, px].set(num_particles)
        self.particles = jnp.column_stack(
            [
                jnp.full(num_particles, px),
                jnp.full(num_particles, py),
                jnp.ones(num_particles),
            ]
        )

        # Track history
        self.history = []

    def random_walk(self, key, particle):
        key1, key2 = random.split(key)
        px, py, opacity = particle

        # Generate movements only for active particles
        delta_x = random.randint(key1, (), -1, 2)
        delta_y = random.randint(key2, (), -1, 2)

        # Update positions and opacity
        new_px = jnp.where(opacity > 0, jnp.clip(px + delta_x, 0, self.m - 1), px)
        new_py = jnp.where(opacity > 0, jnp.clip(py + delta_y, 0, self.m - 1), py)
        new_opacity = jnp.maximum(opacity - 0.1, 0.0)

        return jnp.array([new_px, new_py, new_opacity])

    def simulate(self, num_iterations):
        for _ in range(num_iterations):
            # Generate keys for all particles
            self.master_key, step_key = random.split(self.master_key)
            keys = random.split(step_key, self.num_particles)

            # Update particles
            self.particles = vmap(self.random_walk)(keys, self.particles)

            # Store current state
            self.history.append(self.particles.copy())

            # Update grid (for visualization)
            self.update_grid()

    def update_grid(self):
        # Reset grid
        self.grid = jnp.zeros((self.m, self.m))

        # Accumulate particle opacities in grid cells
        for px, py, opacity in self.particles:
            if opacity > 0:
                self.grid = self.grid.at[int(py), int(px)].add(opacity)

    def get_particle_trajectories(self):
        """Return the full history of particle positions and opacities"""
        return np.array(self.history)

    def get_active_particles(self):
        """Return currently active particles (opacity > 0)"""
        return self.particles[self.particles[:, 2] > 0]
