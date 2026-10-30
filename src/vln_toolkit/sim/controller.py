class ExpertController:
    def reset(self, start_pose: dict) -> None:
        raise NotImplementedError("Initialize the robot pose and controller in Isaac Sim")

    def execute(self, reference_path):
        raise NotImplementedError("Implement physical execution and report measured poses/actions")
