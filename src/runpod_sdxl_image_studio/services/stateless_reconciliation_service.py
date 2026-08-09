"""Startup reconciliation for work restored into a fresh Pod."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

logger = logging.getLogger(__name__)


class StatelessGenerationRepository(Protocol):
    def reconcile_stateless_restore(self, *, now: datetime | None = None) -> int: ...


class StatelessDriveRepository(Protocol):
    def reconcile_stateless_restore(self, now: datetime | None = None) -> int: ...


class StatelessReconciliationService:
    """Apply idempotent failed terminalization before workers start."""

    def __init__(
        self,
        generation_repository: StatelessGenerationRepository,
        drive_repository: StatelessDriveRepository,
        *,
        now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._generation_repository = generation_repository
        self._drive_repository = drive_repository
        self._now_factory = now_factory

    def reconcile(self) -> tuple[int, int]:
        timestamp = self._now_factory()
        generation_count = 0
        drive_count = 0
        try:
            generation_count = self._generation_repository.reconcile_stateless_restore(
                now=timestamp
            )
        except Exception:  # noqa: BLE001 - one subsystem must not block startup
            logger.warning("stateless generation reconciliation failed", exc_info=True)
        try:
            drive_count = self._drive_repository.reconcile_stateless_restore(timestamp)
        except Exception:  # noqa: BLE001 - one subsystem must not block startup
            logger.warning("stateless Drive reconciliation failed", exc_info=True)
        return generation_count, drive_count


__all__ = ["StatelessReconciliationService"]
