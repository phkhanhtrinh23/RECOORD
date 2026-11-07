"""Coordination of repair proposals produced by independent data agents."""

from .model import Atom, Cell, Program, Proposal, Query, Violation, Write
from .coordinator import Coordinator, CoordinatorConfig
from .table import Table

__version__ = "0.2.0"
__all__ = [
    "Atom", "Cell", "Coordinator", "CoordinatorConfig", "Program",
    "Proposal", "Query", "Table", "Violation", "Write",
]
