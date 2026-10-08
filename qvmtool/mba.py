"""A small bitvector expression engine with Mixed Boolean-Arithmetic (MBA) folding.

Purpose
-------
The mutation engine never writes a constant target plainly. It materialises it
through arithmetic that only *evaluates* to the constant, so a naive backward scan
sees an opaque expression. Recovering the target therefore needs two things:

  1. a value model that can carry "known constant" and "known unknown" side by side,
     and
  2. an algebraic simplifier that folds the identities this injector actually emits.

Design notes
------------
* Everything is a fixed 64-bit vector; widths matter because the engine truncates
  through 8/16/32-bit sub-registers, so `BitVec` tracks an explicit width and
  `truncate`/`extend` are first-class.
* `Unknown` is not an error state: it is how a value that genuinely depends on
  program data (a stack slot, a load) is represented, and it is what lets a site be
  reported as *range-constrained* instead of silently mis-resolved.
* Simplification is deliberately a fixed-point rewrite over a small, auditable rule
  set rather than a general SMT query: the rules are exactly the identities observed
  in this engine, and every rule is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

MASK64 = (1 << 64) - 1


def _mask(width: int) -> int:
    return (1 << width) - 1


@dataclass(frozen=True)
class Expr:
    """Base class for expression nodes."""


@dataclass(frozen=True)
class Const(Expr):
    value: int
    width: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & _mask(self.width))


@dataclass(frozen=True)
class Var(Expr):
    """A named source of values: a register, a stack slot, a flag, a load."""

    name: str
    width: int = 64


@dataclass(frozen=True)
class Unknown(Expr):
    """A value that depends on program data; definitively not a constant."""

    reason: str = "data"
    width: int = 64


@dataclass(frozen=True)
class Unary(Expr):
    op: str          # not | neg | bswap | sext | zext | trunc
    a: Expr
    width: int = 64
    arg: int = 0     # byte width for trunc/zext/sext


@dataclass(frozen=True)
class Binary(Expr):
    op: str          # add sub xor and or mul shl shr sar rol ror
    a: Expr
    b: Expr
    width: int = 64


# --------------------------------------------------------------------- helpers

def _is_const(e: Expr, value: Optional[int] = None) -> bool:
    if not isinstance(e, Const):
        return False
    return value is None or e.value == (value & _mask(e.width))


def _c(value: int, width: int = 64) -> Const:
    return Const(value & _mask(width), width)


def _width_of(*es: Expr) -> int:
    for e in es:
        if isinstance(e, (Const, Var, Unknown, Unary, Binary)):
            return e.width
    return 64


# ----------------------------------------------------------------- simplifier

def simplify(e: Expr, rounds: int = 8) -> Expr:
    """Rewrite to a fixed point using the identities this injector emits."""
    cur = e
    for _ in range(rounds):
        nxt = _simplify_once(cur)
        if nxt == cur:
            break
        cur = nxt
    return cur


def _simplify_once(e: Expr) -> Expr:
    if isinstance(e, (Const, Var, Unknown)):
        return e
    if isinstance(e, Unary):
        a = simplify(e.a, rounds=1)
        w = e.width
        if e.op == "not":
            if isinstance(a, Const):
                return _c(~a.value, w)
            if isinstance(a, Unary) and a.op == "not":
                return a.a                      # not(not(x)) == x
        if e.op == "neg":
            if isinstance(a, Const):
                return _c(-a.value, w)
            if isinstance(a, Unary) and a.op == "neg":
                return a.a                      # neg(neg(x)) == x
            if isinstance(a, Unknown):
                return a
        if e.op in ("sext", "zext"):
            if isinstance(a, Const):
                raw = a.value & _mask(e.arg)
                if e.op == "sext" and (raw >> (e.arg * 8 - 1)) & 1:
                    raw |= MASK64 ^ _mask(e.arg * 8)
                return _c(raw, w)
            if isinstance(a, Const):
                return _c(a.value, w)
        if e.op == "trunc":
            if isinstance(a, Const):
                return _c(a.value & _mask(e.arg * 8), w)
            if isinstance(a, Unary) and a.op in ("sext", "zext") \
                    and a.a.width == e.width:
                return a.a                      # trunc(extend(x)) == x
        return Unary(e.op, a, w, e.arg)

    # Binary
    a = simplify(e.a, rounds=1)
    b = simplify(e.b, rounds=1)
    op, w = e.op, e.width
    aw, bw = _width_of(a), _width_of(b)
    both_const = isinstance(a, Const) and isinstance(b, Const)

    # --- constant folding ---
    if both_const:
        x, y = a.value, b.value
        try:
            if op == "add":
                return _c(x + y, w)
            if op == "sub":
                return _c(x - y, w)
            if op == "xor":
                return _c(x ^ y, w)
            if op == "and":
                return _c(x & y, w)
            if op == "or":
                return _c(x | y, w)
            if op == "mul":
                return _c(x * y, w)
            if op == "shl":
                return _c(x << (y & 63), w)
            if op == "shr":
                return _c((x & _mask(w)) >> (y & 63), w)
            if op == "sar":
                v = x & _mask(w)
                if v >> (w - 1):
                    v |= MASK64 ^ _mask(w)
                return _c(v >> (y & 63), w)
            if op == "rol":
                k = y & (w - 1)
                v = x & _mask(w)
                return _c(((v << k) | (v >> (w - k))) if k else v, w)
            if op == "ror":
                k = y & (w - 1)
                v = x & _mask(w)
                return _c(((v >> k) | (v << (w - k))) if k else v, w)
        except (ValueError, OverflowError):
            pass

    # --- absorbing / identity rules (the injector's bread and butter) ---
    if op in ("sub", "xor") and a == b:
        return _c(0, w)                          # x - x == 0 ; x ^ x == 0
    if op in ("and", "or") and a == b:
        return a                                 # x & x == x ; x | x == x
    if op == "sub" and _is_const(b, 0):
        return a
    if op == "add" and _is_const(b, 0):
        return a
    if op in ("xor", "or", "shl", "shr", "sar", "rol", "ror") and _is_const(b, 0):
        return a
    if op == "and" and _is_const(b, 0):
        return _c(0, w)                          # x & 0 == 0
    if op == "mul" and _is_const(b, 1):
        return a
    if op == "mul" and _is_const(b, 0):
        return _c(0, w)
    if op == "and" and _is_const(b, _mask(w)):
        return a
    if op == "or" and _is_const(b, _mask(w)):
        return _c(_mask(w), w)
    if op in ("add", "sub", "xor", "or", "and") and _is_const(a, 0) and op in ("add", "xor", "or"):
        return b
    if op == "sub" and _is_const(a, 0):
        return simplify(Unary("neg", b, w))

    # --- the two algebraic identities that actually appear here ---
    # (x | y) + (x & y) == x + y
    if op == "add" and isinstance(a, Binary) and isinstance(b, Binary) \
            and a.op == "or" and b.op == "and" \
            and {a.a, a.b} == {b.a, b.b}:
        return simplify(Binary("add", a.a, a.b, w))
    # (x | y) - (x & y) == x ^ y
    if op == "sub" and isinstance(a, Binary) and isinstance(b, Binary) \
            and a.op == "or" and b.op == "and" \
            and {a.a, a.b} == {b.a, b.b}:
        return simplify(Binary("xor", a.a, a.b, w))

    # --- unknown propagation ---
    if isinstance(a, Unknown) or isinstance(b, Unknown):
        if op == "and" and _is_const(b, 0):
            return _c(0, w)
        if op == "mul" and _is_const(b, 0):
            return _c(0, w)
        if op in ("add", "sub", "xor", "or") and _is_const(b, 0):
            return a
        return Binary(op, a, b, w)
    if isinstance(a, Unknown) and isinstance(b, Unknown):
        if op in ("sub", "xor") and a.reason == b.reason:
            return _c(0, w)
        return Binary(op, a, b, w)

    if aw != bw and isinstance(a, Const):
        a = _c(a.value, bw)
    return Binary(op, a, b, w)


# ------------------------------------------------------------------ queries

def as_constant(e: Expr) -> Optional[int]:
    s = simplify(e)
    return s.value if isinstance(s, Const) else None


def affine_parts(e: Expr) -> Optional[tuple[int, int]]:
    """If `e` is `constant_base + dynamic`, return (base, dynamic_id).

    The dynamic term does **not** have to be a bare `Unknown`: this engine wraps the
    offset in layers of MBA, so the shape that actually occurs is
    `Const(base) add sext(zext(... Unknown ... rol k ...))`. Requiring a bare
    `Unknown` on the right-hand side (as a first cut did) silently reports those
    sites as opaque and loses the base entirely -- which is the whole point of the
    exercise. So: a constant term plus any term that is not itself constant counts.
    """
    s = simplify(e)
    if isinstance(s, Const):
        return (s.value, 0)
    if isinstance(s, Unknown):
        return (0, hash(s))
    if isinstance(s, Binary) and s.op in ("add", "sub"):
        a, b = simplify(s.a), simplify(s.b)
        a_const = isinstance(a, Const)
        b_const = isinstance(b, Const)
        if a_const and not b_const:
            base = a.value if s.op == "add" else (-a.value)
            return (base & MASK64, hash(b))
        if b_const and not a_const:
            base = b.value if s.op == "add" else (-b.value)
            return (base & MASK64, hash(a))
    # walk down through truncation/extension layers to find a constant head
    if isinstance(s, Unary):
        inner = affine_parts(s.a)
        if inner is not None and inner[1]:
            return inner
    return None


def contains_unknown(e: Expr, reason: Optional[str] = None) -> bool:
    """True if `e` still references an Unknown, optionally one with a given reason.

    This is the test that decides whether a backward slice actually established a
    value or merely started too late: an expression containing `Unknown("reg:rsi")`
    when we are resolving `rsi` means the slice never saw `rsi` written, so the
    verdict must not be trusted.
    """
    s = simplify(e)
    if isinstance(s, Unknown):
        return reason is None or s.reason == reason
    if isinstance(s, Unary):
        return contains_unknown(s.a, reason)
    if isinstance(s, Binary):
        return contains_unknown(s.a, reason) or contains_unknown(s.b, reason)
    return False


def describe(e: Expr, limit: int = 160) -> str:
    """Compact, human-auditable rendering."""
    s = simplify(e)
    out: list[str] = []

    def walk(n: Expr) -> None:
        if isinstance(n, Const):
            out.append(hex(n.value))
        elif isinstance(n, Var):
            out.append(n.name)
        elif isinstance(n, Unknown):
            out.append(f"?{n.reason}")
        elif isinstance(n, Unary):
            out.append(n.op + "(")
            walk(n.a)
            out.append(")")
        elif isinstance(n, Binary):
            out.append("(")
            walk(n.a)
            out.append(f" {n.op} ")
            walk(n.b)
            out.append(")")

    walk(s)
    text = "".join(out)
    return text if len(text) <= limit else text[:limit - 3] + "..."
