"""PE access layer: one place that knows how to turn an RVA into bytes.

Every other module goes through `PEMap`, so section-layout quirks (this family of
binaries relocates the Exception directory into the VM section, and leaves the
`.pdata` section holding encrypted data) are handled once, here.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pefile

from .model import Section

IMAGE_DIR_EXPORT = 0
IMAGE_DIR_IMPORT = 1
IMAGE_DIR_EXCEPTION = 3
IMAGE_DIR_SECURITY = 4
IMAGE_DIR_RELOC = 5
IMAGE_DIR_TLS = 9

#: Section names that mark a virtualisation engine shipped by this vendor.
ENGINE_SECTIONS = {
    ".qvm0": "qvm",
    ".qvm1": "qvm",
    ".tvm0": "tvm",
    ".tvm1": "tvm",
    ".vmp0": "vmp",
    ".vmp1": "vmp",
    ".ruby0": "ruby",
    ".ruby1": "ruby",
    ".ruby2": "ruby",
}


class PEMap:
    def __init__(self, path: str):
        self.path = path
        self.data = open(path, "rb").read()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        self.pe = pefile.PE(data=self.data, fast_load=True)
        self.pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        ])
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.ep = self.pe.OPTIONAL_HEADER.AddressOfEntryPoint
        self.dll_chars = self.pe.OPTIONAL_HEADER.DllCharacteristics
        self.sections: list[Section] = [
            Section(
                name=s.Name.rstrip(b"\0").decode("latin1"),
                va=s.VirtualAddress,
                vsize=s.Misc_VirtualSize,
                raw=s.PointerToRawData,
                rsize=s.SizeOfRawData,
                chars=s.Characteristics,
            )
            for s in self.pe.sections
        ]

    # ---------------- section helpers ----------------

    def section_named(self, name: str) -> Optional[Section]:
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def section_for_rva(self, rva: int) -> Optional[Section]:
        for s in self.sections:
            if s.va <= rva < s.va + s.span:
                return s
        return None

    def section_name_for_rva(self, rva: int) -> Optional[str]:
        s = self.section_for_rva(rva)
        return s.name if s else None

    def exec_sections(self) -> list[Section]:
        return [s for s in self.sections if s.is_exec]

    def vm_sections(self) -> list[Section]:
        return [s for s in self.sections if s.name in ENGINE_SECTIONS]

    # ---------------- data helpers ----------------

    def rva_to_off(self, rva: int) -> Optional[int]:
        s = self.section_for_rva(rva)
        if s is None:
            return None
        off = rva - s.va
        if off < 0 or off >= s.rsize:
            return None
        return s.raw + off

    def read_rva(self, rva: int, size: int) -> Optional[bytes]:
        off = self.rva_to_off(rva)
        if off is None:
            return None
        return self.data[off:off + size]

    def section_bytes(self, section: Section) -> bytes:
        return self.data[section.raw:section.raw + section.rsize]

    def data_directory(self, index: int):
        try:
            return self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[index]
        except IndexError:
            return None

    # ---------------- exports / imports ----------------

    def exports(self) -> list[tuple[str, int, int]]:
        """(name, ordinal, rva); unnamed exports get a synthetic name."""
        out = []
        directory = getattr(self.pe, "DIRECTORY_ENTRY_EXPORT", None)
        if directory is None:
            return out
        for sym in directory.symbols:
            name = sym.name.decode(errors="replace") if sym.name else f"ord{sym.ordinal}"
            out.append((name, sym.ordinal, sym.address))
        return out

    def named_export_count(self) -> int:
        directory = getattr(self.pe, "DIRECTORY_ENTRY_EXPORT", None)
        if directory is None:
            return 0
        return sum(1 for s in directory.symbols if s.name)

    def imports(self) -> list[tuple[str, int]]:
        out = []
        for entry in getattr(self.pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            out.append((entry.dll.decode(errors="replace"), len(entry.imports)))
        return out
