"""Fetching pages from the web and reducing them to readable text.

This is the plumbing under two things: the ``web_search`` tool a chat model can
call, and Teacher's material gathering. It does three jobs and no more —
searching, fetching one page, and stripping markup to text.

It is deliberately not a crawler. Nothing here follows a link found inside a
page; callers pass an address they got from a search or typed themselves.

Addresses are checked before and after redirects, so a public URL that
redirects to ``127.0.0.1`` or a machine on the local network is refused rather
than fetched.

Search goes through DuckDuckGo's HTML endpoint, with Wikipedia's own search
behind it; neither needs a key or an account. Engines get blocked and
redesigned without warning, so when none answers that is said plainly rather
than papered over.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ai_studio.core.errors import ValidationError, WebError

SEARCH_URL = "https://html.duckduckgo.com/html/"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

#: Names the tool and its purpose, behind the "Mozilla/5.0" prefix every client
#: sends. Search engines refuse a bare "compatible; bot" agent outright.
AGENT = "Mozilla/5.0 ai-studio/0.1 (local research assistant)"

#: Politeness and safety limits. A page past this size is almost never prose.
TIMEOUT = 20
MAX_BYTES = 3_000_000
MAX_RESULTS = 25
MAX_REDIRECTS = 5

ALLOWED_SCHEMES = {"http", "https"}
#: Hostnames that would reach this machine or its network rather than the web.
LOCAL_NAMES = re.compile(r"^(localhost|.*\.local|.*\.internal|.*\.localhost)$", re.I)


@dataclass
class Result:
    """One entry from a search page."""

    title: str
    url: str
    snippet: str = ""
    engine: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url,
                "snippet": self.snippet, "engine": self.engine}


@dataclass
class Page:
    """One fetched page, reduced to the text a person would read."""

    url: str
    title: str
    text: str

    @property
    def characters(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "characters": self.characters}


# -------------------------------------------------------------- address checks
def _private(host: str) -> bool:
    """True if this address is on the machine itself or its local network."""
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def safe_url(url: str) -> str:
    """Return ``url`` if it is a public http(s) address, else raise.

    The literal form is checked first, then the addresses the name resolves to,
    so ``http://nothing-to-see.example`` pointing at ``10.0.0.1`` is caught too.

    What this does not stop is a name that resolves differently between this
    check and the connection a moment later. Closing that would mean dialling
    the address directly and carrying the host name separately, which breaks
    certificate checking; against a tool that only visits addresses a person or
    a search engine named, the trade is not worth it.
    """
    parsed = urllib.parse.urlparse((url or "").strip())
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValidationError(
            f"Only http and https addresses are fetched, not {parsed.scheme or 'a bare path'!r}."
        )
    host = parsed.hostname
    if not host:
        raise ValidationError(f"{url!r} has no host.")
    if LOCAL_NAMES.match(host) or _private(host):
        raise ValidationError(f"{host} is this machine or its local network, not the web.")

    try:
        resolved = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        # Cannot resolve it — the fetch will fail with a clearer message than
        # any guess made here.
        return url
    for address in resolved:
        if _private(address):
            raise ValidationError(f"{host} resolves to {address}, which is not on the web.")
    return url


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-check every hop: a public URL may redirect to a private one."""

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_CheckedRedirects())


def fetch_bytes(url: str, *, form: dict[str, str] | None = None) -> bytes:
    """Fetch one address, capped in size and time. Raises WebError on failure.

    ``form`` makes it a POST — the search endpoint answers nothing else.
    """
    request = urllib.request.Request(
        safe_url(url),
        data=urllib.parse.urlencode(form).encode() if form else None,
        headers={
            "User-Agent": AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            "Accept-Language": "en",
        },
    )
    try:
        with _opener.open(request, timeout=TIMEOUT) as response:
            kind = (response.headers.get("Content-Type") or "").lower()
            if kind and not any(word in kind for word in ("html", "text", "xml", "json")):
                raise WebError(f"that is {kind.split(';')[0]}, not a page of text")
            return response.read(MAX_BYTES)
    except urllib.error.HTTPError as exc:
        raise WebError(f"the site answered {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise WebError(f"could not reach it ({exc.reason})") from exc
    except TimeoutError as exc:
        raise WebError(f"it did not answer within {TIMEOUT}s") from exc


# ------------------------------------------------------------------ text from markup
DROP = re.compile(r"<(script|style|nav|header|footer|aside|form|noscript)\b.*?</\1>", re.S | re.I)
BREAKS = re.compile(r"</(p|div|li|h[1-6]|tr)\s*>|<br\s*/?>", re.I)
COMMENT = re.compile(r"<!--.*?-->", re.S)
#: A tag, with quoted attribute values skipped over. The naive ``<[^>]+>`` ends
#: the tag at the first ">" inside an attribute, and real pages are full of
#: them — one Wikipedia infobox spills a page of JSON into the text that way.
TAGS = re.compile(r"""<(?:[^>"']|"[^"]*"|'[^']*')*>""", re.S)

#: Below this a "page" is a cookie wall, a redirect stub or an error screen.
MIN_PAGE_CHARACTERS = 200


def strip_markup(markup: str) -> str:
    """Tags out, entities decoded, whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(TAGS.sub("", COMMENT.sub("", markup)))).strip()


def page_text(markup: str) -> str:
    """Reduce a whole HTML document to its readable text, keeping paragraphs."""
    body = DROP.sub(" ", COMMENT.sub(" ", markup))
    match = re.search(r"<body\b.*?</body>", body, re.S | re.I)
    body = match.group(0) if match else body
    body = BREAKS.sub("\n", body)
    text = html.unescape(TAGS.sub("", body))

    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def page_title(markup: str, fallback: str = "") -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.S | re.I)
    return (strip_markup(match.group(1))[:200] if match else "") or fallback


def fetch_page(url: str) -> Page:
    """Fetch one page as text. Raises ValidationError or WebError; never returns junk."""
    markup = fetch_bytes(url).decode("utf-8", "replace")
    text = page_text(markup)
    if len(text) < MIN_PAGE_CHARACTERS:
        raise WebError("almost no text on the page")
    return Page(url=url, title=page_title(markup, fallback=url), text=text)


# ------------------------------------------------------------------------ search
def unwrap(href: str) -> str:
    """Turn the href on a result into the address it means.

    Two things are in the way: DuckDuckGo sometimes wraps a result in its own
    redirect, and an href in HTML is escaped — ``&amp;`` between query
    parameters makes a URL that 404s.
    """
    href = html.unescape(href.strip())
    if "uddg=" in href:
        target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    if href.startswith("//"):
        return "https:" + href
    return href


#: The anchor that carries a result's title and address. Its attributes come in
#: any order, so the href is pulled out of the tag rather than matched in place.
RESULT_ANCHOR = re.compile(r'<a\b([^>]*\bclass="result__a"[^>]*)>(.*?)</a>', re.S)
HREF = re.compile(r'\bhref="([^"]+)"')
RESULT_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


def parse_results(markup: str, limit: int = 5) -> list[Result]:
    """Pull titles, addresses and snippets out of a DuckDuckGo results page.

    Results are found by their title links, and each one claims the snippet
    between itself and the next link. Keying off the surrounding ``<div>``
    instead looks tidier and breaks the moment a class name changes — and the
    failure is silent: every result ends up wearing the first one's snippet.
    """
    found: list[Result] = []
    seen: set[str] = set()
    anchors = list(RESULT_ANCHOR.finditer(markup))

    for index, anchor in enumerate(anchors):
        href = HREF.search(anchor.group(1))
        if not href:
            continue
        address = unwrap(href.group(1))
        if not address.startswith(("http://", "https://")) or address in seen:
            continue
        seen.add(address)

        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(markup)
        snippet = RESULT_SNIPPET.search(markup, anchor.end(), end)
        found.append(Result(
            title=strip_markup(anchor.group(2))[:200],
            url=address,
            snippet=strip_markup(snippet.group(1))[:300] if snippet else "",
        ))
        if len(found) >= limit:
            break
    return found


def search_web(query: str, *, limit: int = 5) -> list[Result]:
    """The open web, through DuckDuckGo's HTML endpoint. No key, no account.

    It answers a POST and nothing else; a GET returns its ordinary home page.
    """
    markup = fetch_bytes(SEARCH_URL, form={"q": query}).decode("utf-8", "replace")
    return parse_results(markup, limit=limit)


def parse_wikipedia(payload: str, limit: int = 5) -> list[Result]:
    """Turn a MediaWiki search response into results."""
    try:
        hits = json.loads(payload).get("query", {}).get("search", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise WebError(f"Wikipedia answered something unreadable ({exc}).") from exc

    found: list[Result] = []
    for hit in hits[:limit]:
        title = hit.get("title", "")
        if not title:
            continue
        found.append(Result(
            title=title,
            url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            snippet=strip_markup(hit.get("snippet", ""))[:300],
        ))
    return found


def search_wikipedia(query: str, *, limit: int = 5) -> list[Result]:
    """Wikipedia's own search. Narrower than the web, and far more dependable.

    Its text is CC BY-SA: usable, but it comes with conditions, which is why
    every page saved from here keeps its address.
    """
    payload = fetch_bytes(WIKIPEDIA_API + "?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": limit, "format": "json",
    })).decode("utf-8", "replace")
    return parse_wikipedia(payload, limit=limit)


#: Tried in this order when no engine is named. The first that answers wins.
ENGINES = {"web": search_web, "wikipedia": search_wikipedia}


def search(query: str, *, limit: int = 5, engine: str = "auto") -> list[Result]:
    """Ask the web what it has on this. Returns titles, addresses and snippets.

    Engines get blocked, rate-limited and redesigned without warning, so
    ``auto`` tries each in turn rather than depending on one staying up. If
    none answers, that is said plainly — no result is ever invented.
    """
    if not query or not query.strip():
        raise ValidationError("Say what to search for.")
    query = query.strip()
    limit = max(1, min(int(limit), MAX_RESULTS))

    if engine != "auto" and engine not in ENGINES:
        raise ValidationError(
            f"Unknown engine {engine!r}. Use one of: auto, " + ", ".join(ENGINES)
        )
    chosen = list(ENGINES) if engine == "auto" else [engine]

    failures: list[str] = []
    for name in chosen:
        try:
            found = ENGINES[name](query, limit=limit)
        except (WebError, OSError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        if found:
            for result in found:
                result.engine = name
            return found
        failures.append(f"{name}: nothing readable came back")

    raise WebError(
        "The search found nothing.",
        hint="Tried " + "; ".join(failures) + ". A search engine may be "
             "rate-limiting this machine — wait a minute, or name the pages "
             "directly instead of searching.",
    )
