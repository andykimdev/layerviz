"""
Test the inline page helpers that embed text files and reject files they cannot inline.

Purpose:
- Check that a text file such as a license becomes one HTML comment with its text unchanged.
- Check that text able to close the comment early raises instead of leaking into the page.
"""

import pytest

from layerviz.inline import cannot_inline, text_comment


def test_text_comment_wraps_the_text_unchanged_under_its_path():
    assert text_comment("vendor/LICENSE.txt", "MIT -- see below\n") == (
        "<!-- vendor/LICENSE.txt\nMIT -- see below\n\n-->"
    )


@pytest.mark.parametrize("text", ["a --> b", "a --!> b", "-->"])
def test_text_comment_rejects_text_that_closes_the_comment_early(text):
    with pytest.raises(ValueError, match="vendor/LICENSE.txt"):
        text_comment("vendor/LICENSE.txt", text)


def test_text_comment_keeps_a_trailing_dash_pair_apart_from_the_closing_marker():
    assert text_comment("vendor/LICENSE.txt", "ends with --") == (
        "<!-- vendor/LICENSE.txt\nends with --\n-->"
    )


def test_cannot_inline_names_files_other_than_js_css_and_txt():
    web = {"app.js": b"", "style.css": b"", "vendor/LICENSE.txt": b"", "logo.png": b"", "a/b.wasm": b""}
    assert cannot_inline(web) == ["a/b.wasm", "logo.png"]


def test_cannot_inline_empty_web_folder_names_nothing():
    assert cannot_inline({}) == []
