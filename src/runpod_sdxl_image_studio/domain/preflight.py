"""Typed generation preflight results shared by UI and application services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PreflightSeverity(StrEnum):
    """Severity used to decide whether enqueue may proceed."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class PreflightIssue:
    """One safe, user-facing preflight finding."""

    code: str
    message: str
    severity: PreflightSeverity | str

    @property
    def is_error(self) -> bool:
        return self.severity == PreflightSeverity.ERROR or self.severity == "error"

    @property
    def is_warning(self) -> bool:
        return self.severity == PreflightSeverity.WARNING or self.severity == "warning"


@dataclass(frozen=True)
class PreflightResult:
    """Result of the UX preflight performed immediately before enqueue."""

    is_ready: bool
    errors: tuple[PreflightIssue, ...]
    warnings: tuple[PreflightIssue, ...]
    checked_at: datetime

    @property
    def issues(self) -> tuple[PreflightIssue, ...]:
        return self.errors + self.warnings

    @property
    def error_messages(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.errors)

    @property
    def warning_messages(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.warnings)


__all__ = ["PreflightIssue", "PreflightResult", "PreflightSeverity"]
