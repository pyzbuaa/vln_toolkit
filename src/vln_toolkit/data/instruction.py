"""Geometry-only templates: never invent object or room semantics."""

import math

import numpy as np

from vln_toolkit.utils.coordinates import wrap_angle


class TemplateInstructionGenerator:
    def __init__(self, *, language="zh", turn_threshold=math.pi / 18):
        if language not in {"zh", "en"}:
            raise ValueError("Supported template languages: zh, en")
        self.language, self.turn_threshold = language, turn_threshold

    def generate(self, route, task=None, *, start_yaw=None) -> str:
        points = np.asarray(route, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("Expected a finite Nx3 route")
        segments = []
        for delta in np.diff(points, axis=0):
            length = float(np.linalg.norm(delta[:2]))
            if length <= 1e-9:
                continue
            heading = math.atan2(delta[1], delta[0])
            if segments and abs(wrap_angle(heading - segments[-1][0])) < 1e-6:
                segments[-1][1] += length
            else:
                segments.append([heading, length])
        zh = self.language == "zh"
        if not segments:
            return "已到达终点。" if zh else "You are at the goal."
        yaw = (
            (task.start_yaw if task is not None else segments[0][0])
            if start_yaw is None
            else start_yaw
        )
        phrases = []
        for heading, distance in segments:
            turn = wrap_angle(heading - yaw)
            if abs(turn) >= self.turn_threshold:
                degrees = round(abs(math.degrees(turn)))
                direction = (
                    ("左" if turn > 0 else "右") if zh else ("left" if turn > 0 else "right")
                )
                phrases.append(
                    f"向{direction}转{degrees}度" if zh else f"Turn {direction} {degrees} degrees"
                )
            phrases.append(
                f"向前行进{distance:.2f}米" if zh else f"Walk forward {distance:.2f} metres"
            )
            yaw = heading
        phrases.append("到达终点" if zh else "Stop at the goal")
        return "，然后".join(phrases) + "。" if zh else ". ".join(phrases) + "."


def generate_instruction(route, task=None, **kwargs):
    return TemplateInstructionGenerator(**kwargs).generate(route, task)
