import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def initialize_simulation(grid_size, num_particles, seed=0):
    """
    Initialize the simulation environment with a grid and particles.

    Args:
        grid_size (int): Size of the square grid (m x m)
        num_particles (int): Number of particles to simulate
        seed (int): Random seed for reproducibility

    Returns:
        tuple: Grid size, random key, initial grid state, particle array, empty history list
    """
    m = grid_size
    master_key = random.key(seed)

    # Initialize grid and particles
    grid = jnp.zeros((m, m))

    # Set initial position to center of grid
    px = m // 2
    py = m // 2

    # Set initial positions and opacities
    grid = grid.at[py, px].set(num_particles)
    particles = jnp.column_stack(
        [
            jnp.full(num_particles, px),
            jnp.full(num_particles, py),
            jnp.ones(num_particles),
        ]
    )

    # Track history
    history = []

    return m, master_key, grid, particles, history


def random_walk(key, particle, grid_size, particles):
    """
    Perform one step of random walk for a single particle.

    Args:
        key: JAX random key for this step
        particle: Single particle state (x, y, opacity)
        grid_size: Size of the simulation grid
        particles: Array of all particles (unused but kept for vmap compatibility)

    Returns:
        array: Updated particle state [new_x, new_y, new_opacity]
    """
    key1, key2 = random.split(key)
    px, py, opacity = particle

    # Generate movements only for active particles
    delta_x = random.randint(key1, (), -1, 2)
    delta_y = random.randint(key2, (), -1, 2)

    # Update positions and opacity
    new_px = jnp.where(opacity > 0, jnp.clip(px + delta_x, 0, grid_size - 1), px)
    new_py = jnp.where(opacity > 0, jnp.clip(py + delta_y, 0, grid_size - 1), py)
    new_opacity = jnp.maximum(opacity - 0.1, 0.0)

    return jnp.array([new_px, new_py, new_opacity])


def simulate(grid_size, num_particles, num_iterations, master_key, grid, particles):
    """
    Run the complete simulation for all particles over multiple iterations.

    Args:
        grid_size: Size of the simulation grid
        num_particles: Number of particles to simulate
        num_iterations: Number of simulation steps
        master_key: JAX random key
        grid: Initial grid state
        particles: Initial particle states

    Returns:
        tuple: Final particle states, history of all states, final grid state
    """
    history = []
    for _ in range(num_iterations):
        # Generate keys for all particles
        master_key, step_key = random.split(master_key)
        keys = random.split(step_key, num_particles)

        # Update particles
        particles = vmap(random_walk, in_axes=(0, 0, None, None))(
            keys, particles, grid_size, particles
        )

        # Store current state
        history.append(particles.copy())

        # Update grid (for visualization)
        grid = update_grid(particles, grid_size)

    return particles, history, grid


def update_grid(particles, grid_size):
    """
    Update the grid state based on current particle positions and opacities.

    Args:
        particles: Array of current particle states
        grid_size: Size of the simulation grid

    Returns:
        array: Updated grid with accumulated particle opacities
    """
    grid = jnp.zeros((grid_size, grid_size))

    # Accumulate particle opacities in grid cells
    for px, py, opacity in particles:
        if opacity > 0:
            grid = grid.at[int(py), int(px)].add(opacity)

    return grid


def accumulate_trajectories(trajectories, grid_size):
    """
    Create a grid showing the accumulated presence of particles over time.

    Args:
        trajectories: History of all particle states
        grid_size: Size of the simulation grid

    Returns:
        array: Grid showing total accumulated particle presence
    """
    accumulation_grid = np.zeros((grid_size, grid_size))
    for particles in trajectories:
        for px, py, opacity in particles:
            if opacity > 0:
                accumulation_grid[int(py), int(px)] += opacity
    return accumulation_grid


def find_unsafe_radius(accumulation_grid, center):
    """
    Calculate the radius of the unsafe zone based on particle density.

    Args:
        accumulation_grid: Grid of accumulated particle presence
        center: Coordinates of the initial position

    Returns:
        float: Radius of the unsafe zone
    """
    threshold = np.mean(accumulation_grid) + np.std(accumulation_grid)

    # Create binary mask of unsafe zone
    unsafe_mask = accumulation_grid > threshold

    # Find points above threshold
    unsafe_points = np.array(np.where(unsafe_mask)).T  # [y, x] coordinates

    if len(unsafe_points) == 0:
        return 0

    # Calculate distances from center to each unsafe point
    distances = np.sqrt(
        (unsafe_points[:, 1] - center[0]) ** 2  # x distances
        + (unsafe_points[:, 0] - center[1]) ** 2  # y distances
    )

    # Use the maximum distance as the radius
    # Adding 0.5 for a small buffer
    return np.max(distances) + 0.5


def plot_heatmap(accumulation_grid, initial_position, grid_size, figsize=(10, 8)):
    """
    Create visualization of the simulation results.

    Args:
        accumulation_grid: Grid of accumulated particle presence
        initial_position: Starting coordinates of particles
        grid_size: Size of the simulation grid
        figsize: Size of the output figure

    Returns:
        Figure: Matplotlib figure object
    """
    plt.figure(figsize=figsize)

    # Create heatmap of trajectories
    im = plt.imshow(
        accumulation_grid,
        cmap="YlOrRd",
        interpolation="nearest",
        aspect="equal",
    )

    # Add colorbar with percentage formatting
    cbar = plt.colorbar(im, label="Intensitas awan panas", orientation="vertical")

    # Format colorbar ticks as percentages
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{(x / accumulation_grid.max()) * 100:.0f}%")
    )

    # Find and plot unsafe zone circle
    unsafe_radius = find_unsafe_radius(accumulation_grid, initial_position)

    # Add the unsafe zone circle
    unsafe_circle = plt.Circle(
        initial_position,
        radius=unsafe_radius,
        color="red",
        fill=False,
        linestyle="--",
        linewidth=2,
        label=f"Zona Berbahaya (<{unsafe_radius:.2f} tick)",
    )
    plt.gca().add_patch(unsafe_circle)

    # Mark the initial position (heat source)
    initial_x, initial_y = initial_position
    circle = plt.Circle(
        (initial_x, initial_y), radius=0.5, color="white", fill=False, linewidth=2
    )
    plt.gca().add_patch(circle)

    # Add a larger, semi-transparent circle to emphasize the heat source
    glow = plt.Circle((initial_x, initial_y), radius=1, color="yellow", alpha=0.3)
    plt.gca().add_patch(glow)

    ticks = np.arange(grid_size)

    # Calculate relative distances from heat source
    x_labels = [abs(int(tick - initial_x)) for tick in ticks]
    y_labels = [abs(int(tick - initial_y)) for tick in ticks]

    # Set ticks and labels
    plt.xticks(ticks, x_labels)
    plt.yticks(ticks, y_labels)

    # Customize the plot
    plt.xlabel("X Relatif (3 km/tick)")
    plt.ylabel("Y Relatif (3 km/tick)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    # Set proper axis limits
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(grid_size - 0.5, -0.5)  # Invert Y axis to match grid coordinates

    plt.tight_layout()
    return plt.gcf()


def visualize_heat_simulation(
    grid_size=10, num_particles=100, num_iterations=1_000, seed=0
):
    # Initialize simulation
    m, master_key, grid, particles, history = initialize_simulation(
        grid_size, num_particles, seed
    )

    # Run simulation
    particles, history, grid = simulate(
        grid_size, num_particles, num_iterations, master_key, grid, particles
    )

    # Get trajectories
    trajectories = np.array(history)

    # Accumulate particle trajectories
    accumulation_grid = accumulate_trajectories(trajectories, grid_size)

    # Get initial position from first particle in first frame
    initial_position = (int(trajectories[0][0][0]), int(trajectories[0][0][1]))

    # Plot heatmap
    plot_heatmap(accumulation_grid, initial_position, grid_size)
    plt.show()
