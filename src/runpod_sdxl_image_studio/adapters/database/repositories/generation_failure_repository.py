"""失敗状態を一括保存するRepositoryの互換モジュール。"""

from .generation_repository import (
    GenerationFailureRepository,
    GenerationFailureRepositoryProtocol,
    GenerationRepositoryError,
)

__all__ = [
    "GenerationFailureRepository",
    "GenerationFailureRepositoryProtocol",
    "GenerationRepositoryError",
]
