"""Minimal plant boundary shared by MuJoCo and future BaseCyclic."""

from typing import Protocol, runtime_checkable

from controller.state import JointPositionCommand, PlantState


@runtime_checkable
class PlantBackend(Protocol):
    """One command/state exchange per control cycle.

    ``takeover`` returns the first feedback sample. ``exchange`` applies one
    complete command and returns the next feedback sample. The backend owns
    whether that means advancing simulation time or waiting for a hardware
    cyclic reply. ``release`` ends backend-specific control ownership.
    """

    def takeover(self) -> PlantState:
        ...

    def exchange(self, command: JointPositionCommand) -> PlantState:
        ...

    def release(self) -> None:
        ...
