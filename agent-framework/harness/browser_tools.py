from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import urlparse

from agent_framework import tool
from playwright.async_api import Browser, Page, Playwright, async_playwright


_CDP_URL_PATTERN = re.compile(r"wss?://[^\s\"'<>]+")
_LIVE_URL_KEYS = ("liveViewUrl", "live_view_url", "liveUrl", "live_url")


@dataclass(slots=True)
class BrowserSessionInfo:
    """Foundry Toolbox が作成したブラウザセッション情報。"""

    cdp_url: str
    live_view_url: str | None
    raw_response: str


class CdpBrowserController:
    """Foundry Toolbox の CDP URL に Playwright で接続して操作する制御クラス。

    このクラスは LLM に直接見せません。
    LLM には `build_browser_function_tools()` で作る小さな関数ツールだけを渡します。
    """

    def __init__(self, *, toolbox: Any, allowed_domains: tuple[str, ...]) -> None:
        self._toolbox = toolbox
        self._allowed_domains = allowed_domains
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._session_info: BrowserSessionInfo | None = None

    async def close(self) -> None:
        """Playwright とリモートブラウザ接続を閉じる。"""

        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._page = None

    async def start(self) -> dict[str, Any]:
        """Toolbox でブラウザセッションを作り、CDP URL に接続する。"""

        if self._page is not None:
            return await self._page_state("browser session already started")

        session_tool = await self._find_create_session_tool()
        response_text = await self._call_create_session(session_tool)
        self._session_info = _parse_session_info(response_text)

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.connect_over_cdp(
            self._session_info.cdp_url
        )

        context = self._browser.contexts[0] if self._browser.contexts else None
        if context is None:
            context = await self._browser.new_context()
        self._page = context.pages[0] if context.pages else await context.new_page()
        return await self._page_state("browser session started")

    async def goto(self, url: str) -> dict[str, Any]:
        """許可ドメイン内の URL へ移動する。"""

        self._ensure_allowed_url(url)
        page = await self._ensure_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        return await self._page_state("navigated")

    async def snapshot(self, *, max_chars: int = 12_000) -> dict[str, Any]:
        """現在ページの読み取り用スナップショットを返す。"""

        page = await self._ensure_page()
        title = await page.title()
        body_text = await page.locator("body").inner_text(timeout=10_000)
        links = await page.locator("a").evaluate_all(
            """elements => elements.slice(0, 80).map((element, index) => ({
                index,
                text: (element.innerText || element.textContent || '').trim(),
                href: element.href || ''
            }))"""
        )
        buttons = await page.locator("button, input[type=button], input[type=submit]").evaluate_all(
            """elements => elements.slice(0, 80).map((element, index) => ({
                index,
                text: (element.innerText || element.value || element.getAttribute('aria-label') || '').trim(),
                selectorHint: element.id ? `#${element.id}` : element.name ? `[name="${element.name}"]` : ''
            }))"""
        )
        inputs = await page.locator("input, textarea, select").evaluate_all(
            """elements => elements.slice(0, 80).map((element, index) => ({
                index,
                tag: element.tagName.toLowerCase(),
                type: element.getAttribute('type') || '',
                name: element.getAttribute('name') || '',
                id: element.id || '',
                placeholder: element.getAttribute('placeholder') || '',
                ariaLabel: element.getAttribute('aria-label') || ''
            }))"""
        )
        return {
            "status": "ok",
            "url": page.url,
            "title": title,
            "text": _truncate(body_text, max_chars),
            "links": links,
            "buttons": buttons,
            "inputs": inputs,
        }

    async def click(self, selector: str) -> dict[str, Any]:
        """CSS selector または Playwright locator selector をクリックする。"""

        page = await self._ensure_page()
        await page.locator(selector).first.click(timeout=30_000)
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        return await self._page_state("clicked")

    async def fill(self, selector: str, text: str) -> dict[str, Any]:
        """入力欄に文字列を入れる。"""

        page = await self._ensure_page()
        await page.locator(selector).first.fill(text, timeout=30_000)
        return await self._page_state("filled")

    async def press(self, key: str) -> dict[str, Any]:
        """現在フォーカスされている要素へキー入力する。"""

        page = await self._ensure_page()
        await page.keyboard.press(key)
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        return await self._page_state("pressed")

    async def _find_create_session_tool(self) -> dict[str, Any]:
        """Toolbox 内からブラウザセッション作成ツールを検索する。"""

        result = await self._toolbox.call_tool("tool_search", query="browser", limit=10)
        text = _content_text(result)
        payload = json.loads(text)
        tools = payload.get("tools", [])
        for found_tool in tools:
            if found_tool.get("name") == "browser_automation___create_session":
                return found_tool
        raise RuntimeError(f"browser_automation___create_session was not found: {text}")

    async def _call_create_session(self, session_tool: dict[str, Any]) -> str:
        """Toolbox の call_tool を _meta.tools 付きで呼ぶ。"""

        result = await self._toolbox.call_tool(
            "call_tool",
            name=session_tool["name"],
            arguments={},
            _meta={"tools": [session_tool]},
        )
        return _content_text(result)

    async def _ensure_page(self) -> Page:
        """ページ未作成ならブラウザセッションを開始してから Page を返す。"""

        if self._page is None:
            await self.start()
        if self._page is None:
            raise RuntimeError("Browser page was not created")
        return self._page

    async def _page_state(self, message: str) -> dict[str, Any]:
        """操作後に Agent が判断しやすい短い状態情報を返す。"""

        page = await self._ensure_page()
        title = await page.title()
        return {
            "status": "ok",
            "message": message,
            "url": page.url,
            "title": title,
            "live_view_url": self._session_info.live_view_url
            if self._session_info
            else None,
        }

    def _ensure_allowed_url(self, url: str) -> None:
        """許可ドメイン外への移動を止める。"""

        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Only http/https URLs are allowed: {url}")
        if host not in self._allowed_domains:
            raise ValueError(
                f"Domain is not allowed: {host}. Allowed: {', '.join(self._allowed_domains)}"
            )


def build_browser_function_tools(controller: CdpBrowserController) -> list[Any]:
    """Agent に渡すブラウザ操作用 function tools を作る。"""

    @tool(approval_mode="never_require")
    async def browser_start() -> str:
        """ブラウザセッションを開始し、CDP 経由で Playwright 接続する。"""

        return _json(await controller.start())

    @tool(approval_mode="never_require")
    async def browser_goto(
        url: Annotated[str, "URL to open. Must be in the allowed domains."],
    ) -> str:
        """許可された URL に移動する。"""

        return _json(await controller.goto(url))

    @tool(approval_mode="never_require")
    async def browser_snapshot(
        max_chars: Annotated[int, "Maximum body text characters to return."] = 12_000,
    ) -> str:
        """現在ページのテキスト、リンク、ボタン、入力欄を取得する。"""

        return _json(await controller.snapshot(max_chars=max_chars))

    @tool(approval_mode="never_require")
    async def browser_click(
        selector: Annotated[str, "CSS selector or Playwright locator selector."],
    ) -> str:
        """指定した selector の要素をクリックする。"""

        return _json(await controller.click(selector))

    @tool(approval_mode="never_require")
    async def browser_fill(
        selector: Annotated[str, "CSS selector or Playwright locator selector."],
        text: Annotated[str, "Text to fill into the target element."],
    ) -> str:
        """指定した入力欄に文字列を入力する。"""

        return _json(await controller.fill(selector, text))

    @tool(approval_mode="never_require")
    async def browser_press(
        key: Annotated[str, "Keyboard key name, for example Enter or Tab."],
    ) -> str:
        """キーボードのキーを押す。"""

        return _json(await controller.press(key))

    return [
        browser_start,
        browser_goto,
        browser_snapshot,
        browser_click,
        browser_fill,
        browser_press,
    ]


def _parse_session_info(text: str) -> BrowserSessionInfo:
    """Toolbox のレスポンスから CDP URL と Live View URL を取り出す。"""

    cdp_url: str | None = None
    live_view_url: str | None = None
    try:
        payload = json.loads(text)
        cdp_url = _find_value_by_key(payload, ("cdpUrl", "cdp_url", "webSocketUrl", "wsEndpoint"))
        live_view_url = _find_value_by_key(payload, _LIVE_URL_KEYS)
    except json.JSONDecodeError:
        payload = None

    if cdp_url is None:
        match = _CDP_URL_PATTERN.search(text)
        cdp_url = match.group(0) if match else None
    if cdp_url is None:
        raise RuntimeError(f"CDP URL was not found in Toolbox response: {text}")

    if live_view_url is None and isinstance(payload, dict):
        live_view_url = _find_http_url(payload)
    return BrowserSessionInfo(cdp_url=cdp_url, live_view_url=live_view_url, raw_response=text)


def _find_value_by_key(value: Any, keys: tuple[str, ...]) -> str | None:
    """ネストした dict/list から指定キーの文字列値を探す。"""

    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, str) and found:
                return found
        for child in value.values():
            found = _find_value_by_key(child, keys)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_value_by_key(child, keys)
            if found:
                return found
    return None


def _find_http_url(value: Any) -> str | None:
    """Live View 候補として使える http/https URL を探す。"""

    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    if isinstance(value, dict):
        for child in value.values():
            found = _find_http_url(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_http_url(child)
            if found:
                return found
    return None


def _content_text(result: Any) -> str:
    """MCP Content 配列や通常値を文字列に寄せる。"""

    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(str(getattr(item, "text", item)) for item in result)
    return str(result)


def _json(value: dict[str, Any]) -> str:
    """Agent が読みやすい JSON 文字列にする。"""

    return json.dumps(value, ensure_ascii=False, indent=2)


def _truncate(text: str, max_chars: int) -> str:
    """巨大な本文でコンテキストを潰さないように切り詰める。"""

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...[truncated {len(text) - max_chars} chars]"
