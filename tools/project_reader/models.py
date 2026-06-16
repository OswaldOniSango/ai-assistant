"""Data models for local project reading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectFile:
    path: str
    language: str
    content: str
