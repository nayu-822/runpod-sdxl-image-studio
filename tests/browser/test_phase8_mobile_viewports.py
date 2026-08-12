from __future__ import annotations

import os

import pytest

pytest.importorskip("playwright.sync_api")


pytestmark = pytest.mark.skipif(
    not os.getenv("IMAGE_STUDIO_BROWSER_URL"),
    reason="IMAGE_STUDIO_BROWSER_URLで確認対象のGradio URLを指定してください。",
)

VIEWPORTS = (
    (320, 568),
    (375, 812),
    (390, 844),
    (430, 932),
    (768, 1024),
    (1280, 800),
)
TAB_CONTROLS = {
    "生成": ("Positive prompt", "生成をキューへ追加", "実使用Seed（コピー）"),
    "システム": (
        "System Health",
        "Refresh system status",
        "Recent errors",
        "State backup status",
        "Pod Lifecycle",
        "Terminate安全判定を更新",
        "安全ならPodをTerminate",
        "Auto-Terminate状態",
        "今すぐ状態をバックアップ",
    ),
    "キュー": ("キューを更新", "選択ジョブをキャンセル"),
    "LoRA管理": ("ComfyUI一覧と同期", "検索"),
    "履歴": ("履歴を検索", "実使用seed（選択してコピー）"),
    "アップスケール": ("親Generation ID", "アップスケールをキューへ追加"),
    "プリセット": ("Preset検索", "現在設定から保存"),
    "外部metadata": ("metadataを解析", "画像（PNG / WebP）"),
    "同期・設定": ("同期状態を更新", "Manifest再構築を登録"),
    "モデル準備": (
        "Google Driveモデル",
        "Remote一覧を更新",
        "選択モデルをPodへ準備",
        "準備状況を更新",
        "キャンセル",
        "再試行",
    ),
}


def _has_visible_text(page: object, text: str) -> bool:
    locator = page.get_by_text(text, exact=True)  # type: ignore[attr-defined]
    return any(locator.nth(index).is_visible() for index in range(locator.count()))


def _click_tab(page: object, label: str) -> bool:
    page.wait_for_function(  # type: ignore[attr-defined]
        """
        label => Array.from(document.querySelectorAll(
            '.tab-wrapper button, [role="tablist"] button, [role="tab"]'
        )).some(element => element.textContent.trim() === label)
        """,
        arg=label,
    )
    tablist = page.locator("[role='tablist']")  # type: ignore[attr-defined]
    tab = page.get_by_role("tab", name=label, exact=True)  # type: ignore[attr-defined]
    if _click_visible(page, tab):
        return False

    # Gradio collapses the tail of the tab bar into a compact menu at narrow widths.
    # The overflow menu has a stable semantic container in the current Gradio DOM;
    # do not infer its trigger from an empty button label.
    tab_navigation = tablist.locator("xpath=..")  # type: ignore[attr-defined]
    overflow = tab_navigation.locator(".overflow-menu")  # type: ignore[attr-defined]
    if not overflow.count():
        overflow = tab_navigation.locator("[class*='overflow-menu']")  # type: ignore[attr-defined]
    trigger = (
        overflow.locator(":scope > button")
        if overflow.count()
        else tab_navigation.locator("button:has(svg)").first  # type: ignore[attr-defined]
    )
    if _click_visible(page, trigger):
        dropdown = tab_navigation.locator(  # type: ignore[attr-defined]
            ".overflow-dropdown:not(.hide), [class*='overflow-dropdown']:not(.hide)"
        )
        dropdown.wait_for(state="visible")  # type: ignore[attr-defined]
        menu_item = dropdown.get_by_role(  # type: ignore[attr-defined]
            "button", name=label, exact=True
        )
        if _click_visible(page, menu_item):
            dropdown.wait_for(state="hidden")  # type: ignore[attr-defined]
            return True

    text_item = page.get_by_text(label, exact=True)  # type: ignore[attr-defined]
    assert _click_visible(page, text_item), f"tab not found: {label}"
    return False


def _click_visible(page: object, locator: object) -> bool:
    for index in range(locator.count()):  # type: ignore[attr-defined]
        candidate = locator.nth(index)  # type: ignore[attr-defined]
        if candidate.is_visible():
            candidate.wait_for(state="visible")  # type: ignore[attr-defined]
            candidate.click()
            page.wait_for_timeout(100)  # type: ignore[attr-defined]
            return True
    return False


def _assert_no_horizontal_overflow(page: object) -> None:
    assert page.evaluate(  # type: ignore[attr-defined]
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 2"
    )


def _visible_role_tab_count(page: object) -> int:
    tabs = page.get_by_role("tab")  # type: ignore[attr-defined]
    return sum(tabs.nth(index).is_visible() for index in range(tabs.count()))


def test_phase8_mobile_viewports_have_no_horizontal_overflow_or_missing_controls() -> None:
    from playwright.sync_api import sync_playwright

    url = os.environ["IMAGE_STUDIO_BROWSER_URL"]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            for width, height in VIEWPORTS:
                page.set_viewport_size({"width": width, "height": height})
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_selector("button", state="visible")
                _assert_no_horizontal_overflow(page)
                needs_overflow_menu = width == 320 and (
                    _visible_role_tab_count(page) < len(TAB_CONTROLS)
                )
                overflow_menu_used = False

                for tab_label, controls in TAB_CONTROLS.items():
                    overflow_menu_used = _click_tab(page, tab_label) or overflow_menu_used
                    for control in controls:
                        assert _has_visible_text(page, control), (
                            f"{control!r} is not visible in {tab_label!r} at {width}x{height}"
                        )
                    _assert_no_horizontal_overflow(page)
                if needs_overflow_menu:
                    assert overflow_menu_used, "320x568 did not exercise the overflow tab menu"
        finally:
            browser.close()


def test_phase9_system_tab_is_usable_at_all_required_viewports() -> None:
    """Keep the System Health surface visible even when tabs collapse."""

    from playwright.sync_api import sync_playwright

    url = os.environ["IMAGE_STUDIO_BROWSER_URL"]
    system_tab_label = "システム"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            for width, height in VIEWPORTS:
                page.set_viewport_size({"width": width, "height": height})
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_selector("button", state="visible")
                _click_tab(page, system_tab_label)
                for control in ("System Health", "Refresh system status", "Recent errors"):
                    assert _has_visible_text(page, control), (
                        f"{control!r} is not visible in System tab at {width}x{height}"
                    )
                _assert_no_horizontal_overflow(page)
        finally:
            browser.close()
