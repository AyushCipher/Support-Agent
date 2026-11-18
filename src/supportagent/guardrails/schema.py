"""Tiny JSON-Schema subset validator (type, required, enum, min/max, pattern, maxLength, additionalProperties)."""
from __future__ import annotations

import re
from typing import Any

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    errs: list[str] = []
    t = schema.get("type")
    if t:
        py = _TYPES[t]
        if not isinstance(value, py) or (t in ("integer", "number") and isinstance(value, bool)):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: above maximum {schema['maximum']}")
    if t == "array":
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        for i, v in enumerate(value):
            errs += validate(v, schema.get("items", {}), f"{path}[{i}]")
    if t == "object":
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in value:
                errs.append(f"{path}.{r}: required")
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: unexpected parameter")
    return errs
