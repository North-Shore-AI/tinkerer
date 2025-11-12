"""Minimal helpers for parsing CLAIM[...] formatted completions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


CLAIM_LINE_RE = re.compile(r"^CLAIM\[(?P<id>[^\]]+)\]\s*(?:\(Document\s+\d+\))?\s*:\s*(?P<body>.*)$", re.IGNORECASE)
RELATION_LINE_RE = re.compile(
    r"^RELATION\s*[:\-]?\s*(?P<src>\S+)\s+(?P<label>supports|refutes|contrasts)\s+(?P<dst>\S+)",
    re.IGNORECASE,
)


@dataclass
class Claim:
    identifier: str
    text: str


def parse_claim_lines(lines: Iterable[str]) -> Dict[str, Claim]:
    claims: Dict[str, Claim] = {}
    for line in lines:
        match = CLAIM_LINE_RE.match(line.strip())
        if not match:
            continue
        identifier = match.group("id").strip().lower()
        claims[identifier] = Claim(identifier=identifier, text=match.group("body").strip())
    return claims


def parse_relation_line(line: str) -> Tuple[str, str, str] | None:
    match = RELATION_LINE_RE.match(line.strip())
    if not match:
        return None
    return match.group("src"), match.group("label").lower(), match.group("dst")
