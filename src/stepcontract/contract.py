"""Strict intent and the small visible-event state machine."""
from dataclasses import dataclass


class UnsupportedOperation(ValueError):
    """The requested observation is outside the declared supported scope."""


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


@dataclass
class Window:
    accumulation_steps: int
    backward_count: int = 0
    steps: int = 0
    cleared: bool = False

    def clear(self):
        early = self.backward_count > 0
        self.backward_count = 0
        self.cleared = True
        return early

    def backward(self):
        self.backward_count += 1
        self.cleared = False

    def step(self):
        count = self.backward_count
        self.backward_count = 0
        self.steps += 1
        self.cleared = False
        return count
