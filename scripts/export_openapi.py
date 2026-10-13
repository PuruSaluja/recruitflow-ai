"""Write the API's OpenAPI schema to the file the web client's types are generated from.

Usage::

    python scripts/export_openapi.py            # rewrite frontend/openapi.json
    python scripts/export_openapi.py --check    # exit 1 if it is out of date

The schema comes from the app itself, not a running server, so no database is
needed. ``frontend/src/api/schema.ts`` is generated from this file (``npm run
gen:api`` in ``frontend/``); the test suite runs ``--check``, so a change to
the API that leaves the client's types behind fails it. A difference only in
descriptions or validation limits, which the types do not use, is a warning:
CI installs the newest FastAPI and pydantic, and a release that rewords one
should not fail the build.
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


#: Keywords that document a value or limit it without changing its TypeScript
#: type. With them taken out, openapi-typescript generates the same types,
#: comments aside (checked against this API's schema).
NON_TYPE_KEYS = frozenset(
    {
        "title",
        "description",
        "summary",
        "example",
        "examples",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
    }
)

#: Objects keyed by names -- of properties, schemas, paths, response codes,
#: media types, headers -- where a key such as ``title`` is a name, not a keyword.
NAME_MAPS = frozenset({"properties", "schemas", "paths", "responses", "content", "headers"})


def type_view(node: Any, name_map: bool = False) -> Any:
    """The parts of a schema the client's TypeScript types are generated from."""
    if isinstance(node, dict):
        if name_map:
            return {key: type_view(value) for key, value in node.items()}
        return {
            key: type_view(value, key in NAME_MAPS)
            for key, value in node.items()
            if key not in NON_TYPE_KEYS
        }
    if isinstance(node, list):
        return [type_view(item) for item in node]
    return node


def _same_types(committed: str, current: str) -> bool:
    try:
        old = json.loads(committed)
    except ValueError:
        return False
    return type_view(old) == type_view(json.loads(current))


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
        help=(
            "write nothing; exit 1 if the file differs from the current schema in "
            "anything the client's types are generated from"
        ),
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
        if current == schema:
            print(f"{args.out} is current")
            return 0
        if current is not None and _same_types(current, schema):
            # A newer FastAPI or pydantic rewording a description, or describing a
            # limit differently, leaves the client's types as they are.
            print(
                f"warning: {args.out} differs from the API only in descriptions and "
                "validation limits, which the client's types do not use; regenerate it "
                "when convenient",
                file=sys.stderr,
            )
            return 0
        print(
            f"{args.out} is out of date; run scripts/export_openapi.py, "
            "then npm run gen:api in frontend/",
            file=sys.stderr,
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform, as .gitattributes stores it.
    args.out.write_text(schema, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
