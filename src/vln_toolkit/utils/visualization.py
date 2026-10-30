"""Dependency-free SVG output for inspecting maps and reference paths."""

from pathlib import Path


def save_map_svg(occupancy, path, routes=()):
    height, width = occupancy.grid.shape
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    colors = {1: "#263238", 255: "#b0bec5"}
    # Run-length encoding keeps large map previews reasonably small.
    for row in range(height):
        col = 0
        while col < width:
            code, start = int(occupancy.grid[row, col]), col
            while col < width and int(occupancy.grid[row, col]) == code:
                col += 1
            if code in colors:
                elements.append(
                    f'<rect x="{start}" y="{height - row - 1}" width="{col - start}" '
                    f'height="1" fill="{colors[code]}"/>'
                )
    for index, route in enumerate(routes):
        points = [
            (
                (p[0] - occupancy.origin[0]) / occupancy.resolution,
                height - (p[1] - occupancy.origin[1]) / occupancy.resolution,
            )
            for p in route
        ]
        if not points:
            continue
        coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        color = ["#e53935", "#1e88e5", "#8e24aa", "#00897b"][index % 4]
        elements.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="0.18"/>'
        )
        for (x, y), fill in [(points[0], "#43a047"), (points[-1], color)]:
            elements.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="0.3" fill="{fill}"/>')
    elements.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")
