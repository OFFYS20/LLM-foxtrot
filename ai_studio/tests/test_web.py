"""Fetching from the web: what it refuses, and what it makes of what it gets.

Nothing here reaches the network. The engine is fed recorded markup, so these
tests say what the parsing and the address checks do — not whether DuckDuckGo
happened to answer today.
"""

from __future__ import annotations

import json

import pytest

from ai_studio.core.errors import ValidationError, WebError
from ai_studio.data import web


# ------------------------------------------------------- addresses it refuses
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8080/",
        "https://localhost/secret",
        "http://LOCALHOST/secret",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://172.16.0.1/",
        "https://[::1]/",
        "http://169.254.169.254/latest/meta-data/",  # the cloud metadata service
        "http://0.0.0.0/",
        "http://printer.local/",
        "http://wiki.internal/",
    ],
)
def test_addresses_on_this_machine_or_its_network_are_refused(url):
    with pytest.raises(ValidationError):
        web.safe_url(url)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "data:text/html,<b>hi", "javascript:alert(1)",
     "not-a-url", "", "https://", "http:///nohost"],
)
def test_only_http_and_https_are_fetched(url):
    with pytest.raises(ValidationError):
        web.safe_url(url)


def test_a_name_that_resolves_to_a_private_address_is_refused(monkeypatch):
    """The literal form looks public; where it points is what matters."""
    monkeypatch.setattr(
        web.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("10.1.2.3", 0))],
    )
    with pytest.raises(ValidationError, match="10.1.2.3"):
        web.safe_url("https://nothing-to-see.example/")


def test_a_public_name_is_allowed(monkeypatch):
    monkeypatch.setattr(
        web.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert web.safe_url("https://example.com/page") == "https://example.com/page"


def test_an_unresolvable_name_is_left_to_the_fetch_to_report(monkeypatch):
    """Guessing here would replace the real error with a worse one."""
    def refuse(*args, **kwargs):
        raise OSError("Name or service not known")

    monkeypatch.setattr(web.socket, "getaddrinfo", refuse)
    assert web.safe_url("https://nowhere.example/") == "https://nowhere.example/"


def test_a_redirect_to_a_private_address_is_refused():
    """A public URL that redirects inward is the usual way past a host check."""
    handler = web._CheckedRedirects()
    with pytest.raises(ValidationError):
        handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/admin")


# ------------------------------------------------------------- markup to text
PAGE = """
<html><head><title>  The  Keeper &amp; the Lamp </title></head>
<body>
  <script>var tracking = {"a": 1 < 2};</script>
  <style>body { color: red }</style>
  <nav>Home | About</nav>
  <div data-mw='{"parts":[{"wt":"&lt;br/> nonsense"}]}'>Kept text.</div>
  <p>First paragraph.</p>
  <p>Second&nbsp;paragraph.</p>
  <footer>&copy; 2026</footer>
</body></html>
"""


def test_the_title_is_taken_and_tidied():
    assert web.page_title(PAGE) == "The Keeper & the Lamp"


def test_page_title_falls_back_when_there_is_none():
    assert web.page_title("<html><body>hi</body></html>", fallback="http://x/") == "http://x/"


def test_scripts_styles_and_navigation_are_dropped():
    text = web.page_text(PAGE)
    assert "tracking" not in text
    assert "color: red" not in text
    assert "Home | About" not in text
    assert "2026" not in text


def test_the_readable_text_survives_with_its_paragraphs():
    text = web.page_text(PAGE)
    assert "First paragraph." in text
    assert "Second paragraph." in text or "Second paragraph." in text
    assert "First paragraph.\nSecond" in text.replace(" ", " ")


def test_an_attribute_holding_markup_does_not_leak_into_the_text():
    """`<[^>]+>` ends a tag at the first '>' inside an attribute; real pages
    carry whole documents in attributes, and it all spills out as 'text'."""
    text = web.page_text(PAGE)
    assert "nonsense" not in text
    assert "Kept text." in text


def test_comments_are_dropped_even_when_they_contain_tags():
    assert "hidden" not in web.page_text("<body><!-- <p>hidden</p> -->visible</body>")


def test_entities_are_decoded():
    assert web.strip_markup("<b>caf&eacute; &amp; cr&egrave;me</b>") == "café & crème"


# ------------------------------------------------------------ search results
#: Shaped like the real thing: the outer divs carry several class names, the
#: anchor attributes come in either order, and one result has no snippet.
DDG = """
<div class="result results_links web-result">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="https://example.com/one">One &amp; Only</a>
    </h2>
    <a class="result__snippet" href="https://example.com/one">The <b>first</b> result.</a>
  </div>
</div>
<div class="result results_links web-result">
  <div class="links_main links_deep result__body">
    <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftwo" class="result__a">Two</a>
    <a class="result__snippet" href="https://example.org/two">The second result.</a>
  </div>
</div>
<div class="result results_links web-result">
  <div class="links_main links_deep result__body">
    <a class="result__a" href="https://example.net/three">Three</a>
  </div>
</div>
<div class="result results_links web-result">
  <div class="links_main links_deep result__body">
    <a class="result__a" href="https://example.com/one">A duplicate</a>
  </div>
</div>
"""


def test_the_redirect_wrapper_is_unwrapped():
    assert web.unwrap("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fb%3Fc%3D1") \
        == "https://a.test/b?c=1"


def test_a_plain_address_is_left_alone():
    assert web.unwrap("https://a.test/b") == "https://a.test/b"


def test_an_escaped_address_is_decoded():
    """`&amp;` between query parameters is a URL that fetches nothing."""
    assert web.unwrap("https://a.test/x?uid=1&amp;id=2") == "https://a.test/x?uid=1&id=2"


def test_a_protocol_relative_address_gets_a_scheme():
    assert web.unwrap("//a.test/b") == "https://a.test/b"


def test_results_are_read_out_of_the_page():
    found = web.parse_results(DDG, limit=5)
    assert [result.url for result in found] == [
        "https://example.com/one", "https://example.org/two", "https://example.net/three"]
    assert found[0].title == "One & Only"


def test_each_result_keeps_its_own_snippet():
    """Scoping the snippet to the surrounding div fails silently when a class
    name changes: every result then wears the first one's text."""
    found = web.parse_results(DDG, limit=5)
    assert found[0].snippet == "The first result."
    assert found[1].snippet == "The second result."
    assert found[2].snippet == "", "a result with no snippet must not borrow one"


def test_a_title_link_is_found_whatever_order_its_attributes_come_in():
    assert web.parse_results(DDG)[1].title == "Two"


def test_the_same_address_is_not_returned_twice():
    assert len(web.parse_results(DDG, limit=10)) == 3


def test_the_limit_is_honoured():
    assert len(web.parse_results(DDG, limit=1)) == 1


def test_a_page_with_no_results_yields_none():
    assert web.parse_results("<html><body>nothing here</body></html>") == []


WIKI = json.dumps({"query": {"search": [
    {"title": "Lighthouse keeper", "snippet": 'A <span class="searchmatch">keeper</span> tends it'},
    {"title": "Flannan Isles Lighthouse", "snippet": "Three men vanished"},
]}})


def test_wikipedia_results_become_article_addresses():
    found = web.parse_wikipedia(WIKI, limit=5)
    assert found[0].url == "https://en.wikipedia.org/wiki/Lighthouse_keeper"
    assert found[1].url == "https://en.wikipedia.org/wiki/Flannan_Isles_Lighthouse"


def test_wikipedia_snippets_have_their_markup_removed():
    assert web.parse_wikipedia(WIKI)[0].snippet == "A keeper tends it"


def test_an_unreadable_wikipedia_answer_is_reported_not_guessed():
    with pytest.raises(WebError):
        web.parse_wikipedia("<html>rate limited</html>")


# ------------------------------------------------------- choosing an engine
def test_an_empty_query_is_refused_before_anything_leaves_the_machine():
    with pytest.raises(ValidationError):
        web.search("   ")


def test_an_unknown_engine_is_named_in_the_error():
    with pytest.raises(ValidationError, match="wikipedia"):
        web.search("anything", engine="altavista")


def test_the_second_engine_is_tried_when_the_first_is_blocked(monkeypatch):
    def blocked(query, *, limit=5):
        raise WebError("the site answered 429")

    monkeypatch.setitem(web.ENGINES, "web", blocked)
    monkeypatch.setitem(
        web.ENGINES, "wikipedia",
        lambda query, *, limit=5: [web.Result(title="T", url="https://ok.test/")],
    )
    found = web.search("lighthouses")
    assert [result.url for result in found] == ["https://ok.test/"]
    assert found[0].engine == "wikipedia", "the result says which engine answered"


def test_when_every_engine_fails_it_says_so_rather_than_inventing_results(monkeypatch):
    def blocked(query, *, limit=5):
        raise WebError("the site answered 429")

    monkeypatch.setitem(web.ENGINES, "web", blocked)
    monkeypatch.setitem(web.ENGINES, "wikipedia", blocked)
    with pytest.raises(WebError) as raised:
        web.search("lighthouses")
    assert "429" in (raised.value.hint or ""), "the reason each engine gave is kept"


def test_an_engine_that_answers_with_nothing_is_not_treated_as_success(monkeypatch):
    monkeypatch.setitem(web.ENGINES, "web", lambda query, *, limit=5: [])
    monkeypatch.setitem(
        web.ENGINES, "wikipedia",
        lambda query, *, limit=5: [web.Result(title="T", url="https://ok.test/")],
    )
    assert web.search("x")[0].url == "https://ok.test/"


# --------------------------------------------------------------- whole pages
def test_a_page_with_almost_no_text_is_refused(monkeypatch):
    """A cookie wall or a redirect stub is not material."""
    monkeypatch.setattr(web, "fetch_bytes", lambda url, **kw: b"<html><body>hi</body></html>")
    with pytest.raises(WebError, match="almost no text"):
        web.fetch_page("https://example.com/")


def test_a_fetched_page_comes_back_as_text(monkeypatch):
    body = "<html><head><title>Long</title></head><body><p>" + ("word " * 200) + "</p></body></html>"
    monkeypatch.setattr(web, "fetch_bytes", lambda url, **kw: body.encode())
    page = web.fetch_page("https://example.com/")
    assert page.title == "Long"
    assert page.characters > web.MIN_PAGE_CHARACTERS
    assert page.to_dict()["url"] == "https://example.com/"
