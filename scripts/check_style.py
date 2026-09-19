"""House style checks that a linter does not cover.

Run by pre-commit and by CI. Two rules:

**No em dashes or en dashes.** A project style decision, applied to source,
configuration, documentation, prompts and datasets.

**No smart quotes.** They arrive by copying from a word processor and then
break string literals and SQL in ways that are tedious to diagnose.

Usage::

    python scripts/check_style.py                 # check the whole repository
    python scripts/check_style.py path/to/file    # check specific files
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Characters that must not appear, keyed by character, with what to use
#: instead. Written as code points so this file does not fail its own check.
FORBIDDEN: dict[str, str] = {
    chr(0x2014): "em dash, use a comma, a colon or a full stop",
    chr(0x2013): "en dash, use a hyphen or the word to",
    chr(0x2018): "left single quote, use a straight apostrophe",
    chr(0x2019): "right single quote, use a straight apostrophe",
    chr(0x201C): "left double quote, use a straight quote",
    chr(0x201D): "right double quote, use a straight quote",
    chr(0x2026): "ellipsis character, use three full stops",
    chr(0x0000): "null byte",
}

#: Suffixes never scanned, because the content is not text.
BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".zip",
        ".gz",
        ".whl",
        ".pyc",
        ".pyd",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".bin",
        ".safetensors",
    }
)

SKIPPED_NAMES = frozenset({".coverage", "coverage.xml"})

SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "htmlcov",
        "dist",
        "build",
        "data",
        "reports",
        ".hf_cache",
    }
)

MAXIMUM_BYTES = 2_000_000


def iter_repository_files(root: Path) -> list[Path]:
    """Return every file in the repository worth checking."""
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIPPED_DIRECTORIES for part in relative_parts):
            continue
        if path.name in SKIPPED_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        if path.stat().st_size > MAXIMUM_BYTES:
            continue
        found.append(path)
    return found


def check(path: Path) -> list[str]:
    """Return a problem description for each forbidden character found."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for character, explanation in FORBIDDEN.items():
            if character in line:
                column = line.index(character) + 1
                problems.append(f"{path}:{number}:{column}: {explanation}")
    return problems


def main(argv: list[str]) -> int:
    """Check the supplied files, or the whole repository."""
    root = Path(__file__).resolve().parent.parent
    targets = [Path(a) for a in argv] if argv else iter_repository_files(root)

    problems: list[str] = []
    for target in targets:
        problems.extend(check(target))

    if problems:
        print(f"Style check failed with {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems[:50]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more", file=sys.stderr)
        return 1

    print(f"Style check passed over {len(targets)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
