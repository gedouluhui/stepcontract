"""Explicit intent checks for a bounded PyTorch training window."""
from .audit import StepContract
from .contract import UnsupportedOperation

__version__ = '0.1.1'
__all__ = ['StepContract', 'UnsupportedOperation', '__version__']
