"""Explicit imports register built-ins; no plugin discovery or dependency engine."""

BUILDERS: dict[str, type] = {}


def register(name: str):
    def decorator(builder_cls):
        if name in BUILDERS:
            raise ValueError(f"Builder already exists: {name}")
        BUILDERS[name] = builder_cls
        return builder_cls

    return decorator
