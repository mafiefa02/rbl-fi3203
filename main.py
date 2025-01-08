# from src.heat_particle_simulation import HeatParticleSimulation

from src.simulation import visualize_heat_simulation
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run heat particle simulation")
    parser.add_argument(
        "--grid_size", type=int, default=10, help="Size of the simulation grid"
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=100,
        help="Number of particles in the simulation",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000, help="Number of iterations to run"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for initialization"
    )

    # Parse arguments
    args = parser.parse_args()
    print("Starting simulation...")

    # Using function-based approach (from `main.py`)
    visualize_heat_simulation(
        grid_size=args.grid_size,
        num_particles=args.num_particles,
        num_iterations=args.num_iterations,
        seed=args.seed,
    )

    # sim = HeatParticleSimulation(args.grid_size, args.num_particles, seed=args.seed)
    # sim.simulate(args.num_iterations)
    # visualizer = HeatTrajectoryVisualizer(args.grid_size)
    # visualizer.accumulate_trajectories(sim.get_particle_trajectories())
    # visualizer.plot(initial_position=(int(sim.get_particle_trajectories()[0][0][0]), int(sim.get_particle_trajectories()[0][0][1])))
    # plt.show()


if __name__ == "__main__":
    main()
