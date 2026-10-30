from vln_toolkit.scenes.loader import load_scene


class USDAdapter:
    def load(self, path, meta=None):
        return load_scene(path, meta)
