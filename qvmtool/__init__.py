"""qvmtool -- detection and symbol recovery for QVM/TVM virtualised PE images.

Design notes for the reader:
  * Nothing here depends on disassembling from a section start. These images
    desynchronise a linear sweep within a few hundred bytes, so the pipeline is
    anchored on the RUNTIME_FUNCTION map recovered from the (relocated) Exception
    directory, plus on byte-level idiom sites.
  * Detection and analysis are separate verbs, and every verdict carries the
    evidence that produced it.
  * Symbols are emitted for Ghidra and IDA so the recovered structure lands in the
    analyst's tool rather than in a log file.
"""

__version__ = "0.2.0"

from .model import Edge, Evidence, Function, Region, Section, StateCarrier, Symbol, Verdict
from .pe import ENGINE_SECTIONS, PEMap
from .survey import Survey, run

__all__ = [
    "__version__",
    "PEMap", "ENGINE_SECTIONS",
    "Section", "Function", "Edge", "Region", "Symbol", "Evidence", "Verdict",
    "StateCarrier",
    "Survey", "run",
]
