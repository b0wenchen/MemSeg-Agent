"""Static integrity checks for the buildless GitHub Pages website."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for name in ("href", "src"):
            if name in attrs:
                self.references.append(attrs[name])
        if tag == "img":
            self.images.append(attrs)


def test_project_page_links_and_assets():
    docs = Path(__file__).resolve().parents[1] / "docs"
    parser = PageParser()
    parser.feed((docs / "index.html").read_text())
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate HTML IDs"
    for reference in parser.references:
        url = urlparse(reference)
        if not url.scheme and not url.netloc:
            if url.path:
                assert (docs / url.path).is_file(), reference
            elif url.fragment:
                assert url.fragment in parser.ids, reference
    assert all(image.get("alt") for image in parser.images)
    assert {"bibtex", "install-code", "bars", "copy-status"} <= set(parser.ids)
