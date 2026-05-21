"""
Standalone tests for Dynamic Lora Weights (reForge) core logic.

No dependency on Forge Neo modules — tests the parsing and weight
handling that was causing crashes.

Run:  python3 test_loractl.py
"""

import re
import sys

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# ── Inline copies of the extension's core functions ───────────────────
import numpy as np


def normalise_steps(step, n_steps):
    if step > 1:
        return float(step)
    if step <= 0:
        return 0.0
    return n_steps * step


def sorted_positions(raw_steps, n_steps):
    steps = [[float(s.strip()) for s in re.split("[@~]", x)]
             for x in re.split("[,;]", str(raw_steps))]
    step_triggers = {}
    if len(steps[0]) == 1:
        step_triggers[0] = steps[0][0]
    else:
        for s in sorted(steps, key=lambda s: normalise_steps(s[1] if len(s) == 2 else 1, n_steps)):
            step_triggers[int(normalise_steps(s[1] if len(s) == 2 else 1, n_steps))] = s[0]
    return step_triggers


def _safe_float(s, fallback=1.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return fallback


def _is_dynamic(s):
    try:
        float(s)
        return False
    except (ValueError, TypeError):
        return '@' in str(s)


def _is_simple_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

pass_count = 0
fail_count = 0


def test(name, fn):
    global pass_count, fail_count
    try:
        fn()
        pass_count += 1
        print(f"  ✓ {name}")
    except AssertionError as e:
        fail_count += 1
        msg = _ANSI_RE.sub('', str(e)) if str(e) else "assertion failed"
        print(f"  ✗ {name}: {msg}")
    except Exception as e:
        fail_count += 1
        print(f"  ✗ {name}: {type(e).__name__}: {e}")


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"expected {b!r}, got {a!r}" + (f" — {msg}" if msg else ""))


def assert_in(k, d, msg=""):
    if k not in d:
        raise AssertionError(f"{k!r} not in {d!r}" + (f" — {msg}" if msg else ""))

# ── sorted_positions ────────────────────────────────────────────────


test("simple float weight", lambda: assert_eq(
    sorted_positions("0.5", 20), {0: 0.5}))

test("single relative step", lambda: assert_eq(
    sorted_positions("0.8@0.5", 20), {10: 0.8}))

test("multiple steps (comma)", lambda: (lambda r: (
    assert_eq(r[0], 0.0),
    assert_eq(r[8], 0.5),
    assert_eq(r[20], 1.0),
))(sorted_positions("0@0,0.5@0.4,1@1", 20)))

test("multiple steps (semicolon)", lambda: (lambda r: (
    assert_eq(r[0], 0.0),
    assert_eq(r[10], 0.5),
    assert_eq(r[20], 1.0),
))(sorted_positions("0@0; 0.5@0.5; 1@1", 20)))

test("mixed delimiters", lambda: (lambda r: (
    assert_eq(r[0], 0.0),
    assert_eq(r[10], 0.5),
    assert_eq(r[20], 1.0),
))(sorted_positions("0@0, 0.5@0.5; 1@1", 20)))

test("absolute steps (2.5@5)", lambda: (
    assert_eq(sorted_positions("1.2@5,0.8@10", 20), {5: 1.2, 10: 0.8})
))

test("zero steps", lambda: assert_eq(
    sorted_positions("0.5", 0), {0: 0.5}))

# ── normalise_steps ─────────────────────────────────────────────────


test("relative 0.5 → half", lambda: assert_eq(
    normalise_steps(0.5, 20), 10.0))

test("absolute (>1) unchanged", lambda: assert_eq(
    normalise_steps(1.5, 20), 1.5))

test("@1 → last step (20)", lambda: assert_eq(
    normalise_steps(1.0, 20), 20.0))

test("negative → 0", lambda: assert_eq(
    normalise_steps(-0.1, 20), 0.0))

test("step 0 → 0", lambda: assert_eq(
    normalise_steps(0.0, 20), 0.0))

# ── _is_dynamic ─────────────────────────────────────────────────────


test("plain float → False", lambda: assert_eq(
    _is_dynamic("0.5"), False))

test("complex comma → True", lambda: assert_eq(
    _is_dynamic("0@0, 1@1"), True))

test("complex semicolon → True", lambda: assert_eq(
    _is_dynamic("0@0; 1@1"), True))

test("numeric input → False", lambda: assert_eq(
    _is_dynamic(0.5), False))

test("invalid with @ → True", lambda: assert_eq(
    _is_dynamic("abc@0"), True))

test("invalid without @ → False", lambda: assert_eq(
    _is_dynamic("abc"), False))

# ── _safe_float ─────────────────────────────────────────────────────


test("normal float string", lambda: assert_eq(
    _safe_float("0.5"), 0.5))

test("complex → fallback 1.0", lambda: assert_eq(
    _safe_float("0@0, 1@1"), 1.0))

test("invalid → fallback 1.0", lambda: assert_eq(
    _safe_float("abc"), 1.0))

test("numeric input", lambda: assert_eq(
    _safe_float(0.5), 0.5))

test("zero string", lambda: assert_eq(
    _safe_float("0"), 0.0))

test("negative string", lambda: assert_eq(
    _safe_float("-0.5"), -0.5))

# ── User's actual weight strings ────────────────────────────────────


test("USER: 0.2@0, 0.2@0.4, 1@1", lambda: (lambda r: (
    assert_eq(r[0], 0.2),
    assert_eq(r[8], 0.2),
    assert_eq(r[20], 1.0),
))(sorted_positions("0.2@0, 0.2@0.4, 1@1", 20)))

test("USER: 0@0, 0@0.4, 1@0.5", lambda: (lambda r: (
    assert_eq(r[0], 0.0),
    assert_eq(r[8], 0.0),
    assert_eq(r[10], 1.0),
))(sorted_positions("0@0, 0@0.4, 1@0.5", 20)))

test("USER: 0@0, 0@0.4, 1@0.5 (semicolons)", lambda: (lambda r: (
    assert_eq(r[0], 0.0),
    assert_eq(r[8], 0.0),
    assert_eq(r[10], 1.0),
))(sorted_positions("0@0; 0@0.4; 1@0.5", 20)))

test("USER: 0.2@0, 0.2@0.4, 1@1 is dynamic", lambda: assert_eq(
    _is_dynamic("0.2@0, 0.2@0.4, 1@1"), True))

test("USER: safe_float falls back", lambda: assert_eq(
    _safe_float("0.2@0, 0.2@0.4, 1@1"), 1.0))

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  {pass_count} passed, {fail_count} failed out of {pass_count + fail_count} tests")
if fail_count:
    print("  ❌ SOME TESTS FAILED")
    sys.exit(1)
else:
    print("  ✅ ALL TESTS PASSED")
