"""Immutable result records shared across the QVM tool.

Keeping these as plain dataclasses (not dicts) means every stage of the pipeline
speaks the same vocabulary and the JSON emitters stay mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class Section:
    name: str
    va: int
    vsize: int
    raw: int
    rsize: int
    chars: int

    @property
    def is_exec(self) -> bool:
        return bool(self.chars & 0x20000000)

    @property
    def is_write(self) -> bool:
        return bool(self.chars & 0x80000000)

    @property
    def span(self) -> int:
        return max(self.vsize, self.rsize)


@dataclass(frozen=True)
class Function:
    """A RUNTIME_FUNCTION entry recovered from the image."""

    begin: int            # RVA
    end: int              # RVA
    unwind: int           # RVA of UNWIND_INFO
    section: Optional[str] = None

    @property
    def size(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class Edge:
    """A validated control-transfer from native code into the VM section."""

    caller: int           # RVA of the owning function entry
    site: int             # RVA of the branching instruction
    kind: str             # call | jmp
    target: int           # RVA inside the VM section
    via_resync: bool = False   # decoded only after resynchronising past bad bytes


@dataclass(frozen=True)
class Region:
    section: str
    start: int
    end: int
    kind: str             # code | data | encrypted | sparse | unwind
    entropy: float

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class Symbol:
    address: int
    name: str
    kind: str             # export | function | vm_stub | vm_entry | label
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    claim: str
    detail: str
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    engine: str = "none"          # qvm | tvm | vmp | ruby | none
    confidence: float = 0.0
    vm_section: Optional[str] = None
    vm_range: Optional[tuple[int, int]] = None
    #: which *core mechanism* was actually located, independently of the section
    #: name: qvm-mutation | tvm-threaded | section-only | none. A `.tvm0` section
    #: without a VIP slot and without a fetch is a decoy or a different variant, and
    #: saying so is more useful than folding it into the same verdict.
    engine_core: str = "none"
    evidence: list[Evidence] = field(default_factory=list)

    def add(self, claim: str, detail: str, weight: float = 0.0) -> None:
        self.evidence.append(Evidence(claim, detail, weight))
        self.confidence = min(1.0, self.confidence + weight)


@dataclass
class StateCarrier:
    """Where the virtual machine keeps its architectural state."""

    carrier: str = "unknown"      # stack | register | context_heap | mixed | unknown
    stack_slot_refs: int = 0
    obfuscated_index_refs: int = 0
    prologue_pushes: int = 0
    register_pressure: int = 0
    heap_context_refs: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
