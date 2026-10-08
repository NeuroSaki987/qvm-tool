"""Engine profiles: the parts of the model that are NOT shared by `.qvm0` / `.tvm0`.

`qvmtool` grew up on `.qvm0`, and several stages silently encode QVM facts as if
they were facts about "the vendor's VM":

  * the state sample anchors on `call $+5` sites        (QVM: 22,777 of them; TVM: 3)
  * "no consecutive pointer run" is read as "no opcode fetch" (TVM has a fetch)
  * "`.pdata` encrypted" is treated as core evidence    (TVM `.pdata` is plaintext)
  * native -> VM entries are assumed to be `call rel32`  (TVM: `E9 rel32` trampolines)

Rather than sprinkle `if engine == ...` through every module, each engine gets a
small immutable profile and a stage asks the profile what to do. A profile is
descriptive, not a switch on a name: `detect` fills one in from measured evidence.

The distinction this file makes explicit is between *shared* structure (a relocated
Exception directory, a VM section with no handler table, MBA mutation) and
*engine-specific* structure (how control reaches the VM, where the VIP lives, how
wide a bytecode word is, whether a fetch exists at all).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .pe import ENGINE_SECTIONS


@dataclass(frozen=True)
class EngineProfile:
    name: str                       # qvm | tvm | vmp | ruby | unknown
    section_names: tuple[str, ...]
    #: what the *`.pdata` section* is expected to hold, as a diagnostic expectation
    pdata_expectation: str          # encrypted | stale-plaintext | unknown
    #: how native code reaches the VM payload
    native_entry_form: str          # call-rel32 | e9-trampoline | mixed | unknown
    #: control transfer inside the VM payload
    dispatch_model: str             # obfuscated-constant-branches | threaded-handlers
    #: is the get-PC idiom the engine's own core mechanism?
    self_pc_idiom: str              # engine-core | incidental
    #: where the MBA sample should anchor; each name maps to a provider in
    #: `survey.sample_anchors`
    sample_anchors: tuple[str, ...] = ("call-self-pc",)
    #: RBP-relative byte offset of the VM instruction pointer slot, if the engine
    #: keeps one. Discovered, never assumed -- see `tvm.discover_vip_slot`.
    vip_slot_offset: Optional[int] = None
    #: width in bytes of one VM instruction word, if the engine fetches one
    bytecode_unit: Optional[int] = None
    has_handler_table: bool = False
    #: extra signals `detect` should look for, by name
    core_signals: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


QVM = EngineProfile(
    name="qvm",
    section_names=(".qvm0", ".qvm1"),
    pdata_expectation="encrypted",
    native_entry_form="call-rel32",
    dispatch_model="obfuscated-constant-branches",
    self_pc_idiom="engine-core",
    sample_anchors=("call-self-pc",),
    vip_slot_offset=None,
    bytecode_unit=None,
    has_handler_table=False,
    core_signals=("self-pc-density",),
    notes=(
        "MBA-mutated native code, not an interpreter: no bytecode fetch was observed "
        "in the reference sample (0 data reads of the VM section).",
        "the `call $+5; ...; add reg,imm; jmp reg` idiom is anti-disassembly padding "
        "whose target is the fall-through, so it is CFG-neutral by construction.",
        "state is addressed on the stack by a computed index (VIP lives in the frame).",
    ),
)

TVM = EngineProfile(
    name="tvm",
    section_names=(".tvm0", ".tvm1"),
    pdata_expectation="stale-plaintext",
    native_entry_form="e9-trampoline",
    dispatch_model="threaded-handlers",
    self_pc_idiom="incidental",
    sample_anchors=("threaded-epilogue", "vip-load"),
    vip_slot_offset=None,          # discovered per image
    bytecode_unit=2,               # one VM word is 16 bits
    has_handler_table=False,
    core_signals=("vip-slot-fetch", "threaded-epilogue", "e9-trampoline"),
    notes=(
        "a real threaded interpreter: each handler ends in an inline dispatch "
        "epilogue, so there is still no handler table even though there IS a fetch.",
        "the VIP is a memory slot in the RBP-relative VM context, reloaded by every "
        "handler; the scratch register differs per module, the slot offset does not.",
        "`.pdata` keeps a stale plaintext RUNTIME_FUNCTION[] that is a strict subset "
        "of the live table in the Exception directory.",
    ),
)

UNKNOWN = EngineProfile(
    name="unknown",
    section_names=(),
    pdata_expectation="unknown",
    native_entry_form="unknown",
    dispatch_model="unknown",
    self_pc_idiom="incidental",
    sample_anchors=("call-self-pc",),
    notes=("no engine profile matched; stages fall back to the QVM-era behaviour.",),
)

_BY_NAME = {"qvm": QVM, "tvm": TVM, "vmp": UNKNOWN, "ruby": UNKNOWN}


def profile_for_section(section_name: Optional[str]) -> EngineProfile:
    """Profile implied by a section name (nothing more than a name match)."""
    if not section_name:
        return UNKNOWN
    return _BY_NAME.get(ENGINE_SECTIONS.get(section_name, "unknown"), UNKNOWN)


def profile_for_engine(engine: str) -> EngineProfile:
    return _BY_NAME.get(engine, UNKNOWN)


def all_profiles() -> tuple[EngineProfile, ...]:
    return (QVM, TVM)
