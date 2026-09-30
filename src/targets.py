"""Posterior targets and their coordinate indices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    dimensions: tuple[int, ...]


TARGETS = {
    "mu": Target("mu", (0,)),
    "joint_log": Target("joint_log", (0, 1)),
}
