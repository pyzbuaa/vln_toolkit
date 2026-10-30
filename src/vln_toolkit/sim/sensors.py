class RGBDSensor:
    """Future adapter: frames use optical X-right/Y-down/Z-forward; depth is metres."""

    def capture(self) -> dict:
        raise NotImplementedError("Connect an Isaac Sim camera and return timestamped RGB/depth")
