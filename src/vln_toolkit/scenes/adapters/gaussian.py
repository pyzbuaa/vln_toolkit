class GaussianAdapter:
    def load(self, visual_asset, collision_geometry=None, config=None):
        raise NotImplementedError(
            "3DGS adaptation requires a version-specific renderer and coordinate conversion. "
            "Export a compatible USD first; record visual and collision assets separately."
        )
