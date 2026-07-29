"""生成永続化失敗の安全で安定した分類。"""

from __future__ import annotations

from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepositoryError,
)


class GenerationPersistenceError(GenerationRepositoryError):
    """生成履歴の永続化処理に関する基底例外。"""


class PromptPersistenceError(GenerationPersistenceError):
    """ComfyUI prompt IDをGenerationへ関連付けできない場合の例外。"""


class ArtifactPersistenceError(GenerationPersistenceError):
    """主画像Artifactを登録できない場合の例外。"""


class CompletionPersistenceError(GenerationPersistenceError):
    """GenerationとJobの完了状態を確定できない場合の例外。"""


class RecoveryPersistenceError(GenerationPersistenceError):
    """復旧状態を確定できない場合の例外。"""


class FailurePersistenceError(GenerationPersistenceError):
    """GenerationとJobの失敗状態を確定できない場合の例外。"""


_ERROR_DETAILS: tuple[type[GenerationPersistenceError], str, str] = (
    GenerationPersistenceError,
    "database_error",
    "生成履歴を保存できませんでした。",
)


def persistence_error_code(error: GenerationPersistenceError) -> str:
    """永続化エラーに対応する検索可能な固定コードを返す。"""

    mapping: tuple[tuple[type[GenerationPersistenceError], str], ...] = (
        (PromptPersistenceError, "prompt_persistence_error"),
        (ArtifactPersistenceError, "artifact_persistence_error"),
        (CompletionPersistenceError, "completion_persistence_error"),
        (RecoveryPersistenceError, "recovery_persistence_error"),
        (FailurePersistenceError, "failure_persistence_error"),
    )
    for error_type, code in mapping:
        if isinstance(error, error_type):
            return code
    return _ERROR_DETAILS[1]


def persistence_error_message(error: GenerationPersistenceError) -> str:
    """実装詳細を漏らさないユーザー向けメッセージを返す。"""

    mapping: tuple[tuple[type[GenerationPersistenceError], str], ...] = (
        (
            PromptPersistenceError,
            "生成要求はComfyUIへ送信されましたが、履歴へ関連付けできませんでした。"
            "同じ生成要求の再送信は行っていません。",
        ),
        (
            ArtifactPersistenceError,
            "画像は保存されましたが、履歴へ画像情報を登録できませんでした。",
        ),
        (
            CompletionPersistenceError,
            "画像は保存されましたが、履歴の完了状態を確定できませんでした。",
        ),
        (
            RecoveryPersistenceError,
            "未完了生成の結果は確認できましたが、履歴の復旧状態を保存できませんでした。",
        ),
        (
            FailurePersistenceError,
            "生成は失敗しましたが、履歴の失敗状態を完全に保存できませんでした。",
        ),
    )
    for error_type, message in mapping:
        if isinstance(error, error_type):
            return message
    return _ERROR_DETAILS[2]
