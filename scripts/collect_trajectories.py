import argparse

from vln_toolkit.data import KinematicExpert, PhysicalExpert, TrajectoryRecorder, load_episodes


def main():
    parser = argparse.ArgumentParser(description="Replay reference paths in ideal kinematic mode")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", help="Defaults to the input dataset directory")
    parser.add_argument("--mode", choices=["kinematic", "physical"], default="kinematic")
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()
    expert = (
        KinematicExpert(step_size=args.step_size, dt=args.dt)
        if args.mode == "kinematic"
        else PhysicalExpert()
    )
    recorder = TrajectoryRecorder(args.output or args.dataset)
    episodes = load_episodes(args.dataset)
    for episode in episodes:
        recorder.save(episode.episode_id, expert.execute(episode))
    print(
        f"Saved {len(episodes)} {args.mode} trajectories (kinematic replay has no sensor observations)"
    )


if __name__ == "__main__":
    main()
