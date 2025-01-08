import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class HeatTrajectoryVisualizer:
    def __init__(self, grid_size):
        self.m = grid_size
        self.accumulation_grid = np.zeros((grid_size, grid_size))

    def accumulate_trajectories(self, trajectories):
        """Accumulate particle positions weighted by their opacity"""
        for particles in trajectories:
            for px, py, opacity in particles:
                if opacity > 0:
                    self.accumulation_grid[int(py), int(px)] += opacity

    def find_unsafe_radius(self, center):
        """Find the approximate radius of the unsafe zone based on statistical threshold"""
        threshold = np.mean(self.accumulation_grid) + np.std(self.accumulation_grid)

        # Create binary mask of unsafe zone
        unsafe_mask = self.accumulation_grid > threshold

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

    def plot(self, initial_position, figsize=(10, 8)):
        """Plot the heatmap with the initial position marked and circular safe zone"""
        plt.figure(figsize=figsize)

        # Create heatmap of trajectories
        im = plt.imshow(
            self.accumulation_grid,
            cmap="YlOrRd",
            interpolation="nearest",
            aspect="equal",
        )

        # Add colorbar with percentage formatting
        cbar = plt.colorbar(im, label="Intensitas awan panas", orientation="vertical")

        # Format colorbar ticks as percentages
        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{(x / self.accumulation_grid.max()) * 100:.0f}%"
            )
        )

        # Find and plot unsafe zone circle
        unsafe_radius = self.find_unsafe_radius(initial_position)

        # Add the unsafe zone circle
        unsafe_circle = plt.Circle(
            initial_position,
            radius=unsafe_radius,
            color="red",
            fill=False,
            linestyle="--",
            linewidth=2,
            label=f"Zona Berbahaya (<{unsafe_radius:.2f})",
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

        ticks = np.arange(self.m)

        # Calculate relative distances from heat source
        x_labels = [abs(int(tick - initial_x)) for tick in ticks]
        y_labels = [abs(int(tick - initial_y)) for tick in ticks]

        # Set ticks and labels
        plt.xticks(ticks, x_labels)
        plt.yticks(ticks, y_labels)

        # Customize the plot
        # plt.title('Heat Particle Trajectories with Circular Safe Zone')
        plt.xlabel("X Relatif (3 km/tick)")
        plt.ylabel("Y Relatif (3 km/tick)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()

        # Set proper axis limits
        plt.xlim(-0.5, self.m - 0.5)
        plt.ylim(self.m - 0.5, -0.5)  # Invert Y axis to match grid coordinates

        plt.tight_layout()
        return plt.gcf()
