#!/usr/bin/env python3
"""
Helpers for parsing, enforcing, and rendering CLAIM/RELATION formatted outputs.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


CLAIM_LINE_RE = re.compile(r"^CLAIM\[(?P<id>[^\]]+)\]\s*:\s*(?P<body>.*)$", re.IGNORECASE)
RELATION_LINE_RE = re.compile(
    r"^RELATION\s*[:\-]?\s*(?P<src>\S+)\s+(?P<label>supports|refutes|contrasts)\s+(?P<dst>\S+)",
    re.IGNORECASE,
)


def parse_claim_lines(lines: Iterable[str]) -> List[Dict[str, str]]:
    claims: List[Dict[str, str]] = []
    for line in lines:
        match = CLAIM_LINE_RE.match(line.strip())
        if not match:
            continue
        claims.append(
            {"id": match.group("id").strip(), "text": match.group("body").strip()}
        )
    return claims


def render_claim_lines(claims: Iterable[Dict[str, str]]) -> List[str]:
    return [f"CLAIM[{entry['id']}]: {entry['text']}" for entry in claims]


def enforce_c1(claims: List[Dict[str, str]], gold_text: str) -> List[Dict[str, str]]:
    canonical = gold_text.strip()
    updated: List[Dict[str, str]] = []
    c1_found = False
    for entry in claims:
        cid = entry["id"]
        text = entry["text"]
        if cid.lower() == "c1":
            text = canonical
            c1_found = True
        updated.append({"id": cid, "text": text})
    if not c1_found:
        updated.insert(0, {"id": "c1", "text": canonical})
    return updated


def parse_relation_line(line: str) -> Tuple[str, str, str] | None:
    match = RELATION_LINE_RE.match(line.strip())
    if not match:
        return None
    return match.group("src"), match.group("label").lower(), match.group("dst")


def claims_to_dict(claims: List[Dict[str, str]]) -> Dict[str, str]:
    return {entry["id"].lower(): entry["text"] for entry in claims}


def replace_claim_text(claims: List[Dict[str, str]], claim_id: str, new_text: str) -> None:
    for entry in claims:
        if entry["id"].lower() == claim_id.lower():
            entry["text"] = new_text
            break
