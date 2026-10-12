"""Write the API's OpenAPI schema to the file the web client's types are generated from.

Usage::

    python scripts/export_openapi.py            # rewrite frontend/openapi.json
    python scripts/export_openapi.py --check    # exit 1 if it is out of date

The schema comes from the app itself, not a running server, so no database is
needed. ``frontend/src/api/schema.ts`` is generated from this file (``npm run
gen:api`` in ``frontend/``); the test suite runs ``--check``, so a change to
the API that leaves the client's types behind fails it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT
from app.main import create_app

DEFAULT_OUT = REPO_ROOT / "frontend" / "openapi.json"


def without_patterns(node: Any) -> Any:
    """The schema with every validation ``pattern`` left out.

    The client's types do not use them, and they vary with the pydantic
    version: 2.13 describes a decimal field's digits with a regular
    expression and 2.14 does not, so keeping them made the committed file
    match one installed version only. A property *named* ``pattern`` is kept:
    its value is a schema, not a string.
    """
    if isinstance(node, dict):
        return {
            key: without_patterns(value)
            for key, value in node.items()
            if not (key == "pattern" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [without_patterns(item) for item in node]
    return node


def render_schema() -> str:
    """The schema as the committed file holds it: indented, ASCII, one trailing newline."""
    schema = without_patterns(create_app().openapi())
    return json.dumps(schema, indent=2, ensure_ascii=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if the file differs from the current schema",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = render_schema()
    if args.check:
        try:
            current = args.out.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = None
        if current != schema:
            print(
                f"{args.out} is out of date; run scripts/export_openapi.py, "
                "then npm run gen:api in frontend/",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is current")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the file is compared byte for byte, and Windows would write \r\n.
    args.out.write_text(schema, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
