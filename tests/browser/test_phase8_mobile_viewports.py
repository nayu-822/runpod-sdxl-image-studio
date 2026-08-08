from __future__ import annotations

import os

import pytest

pytest.importorskip("playwright.sync_api")


pytestmark = pytest.mark.skipif(
    not os.getenv("IMAGE_STUDIO_BROWSER_URL"),
    reason="IMAGE_STUDIO_BROWSER_URLを指定した実アプリ起動時だけ実行する",
)


def test_phase8_mobile_viewports_have_no_horizontal_overflow() -> None:
    from playwright.sync_api import sync_playwright

    url = os.environ["IMAGE_STUDIO_BROWSER_URL"]
    viewports = (320, 375, 390, 430, 768, 1024)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            for width in viewports:
                page.set_viewport_size({"width": width, "height": 900})
                page.goto(url, wait_until="networkidle")
                assert page.evaluate(
                    "document.documentElement.scrollWidth "
                    "<= document.documentElement.clientWidth + 2"
                )
        finally:
            browser.close()
