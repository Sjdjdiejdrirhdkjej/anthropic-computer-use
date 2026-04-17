"""Tests for computer_use_demo.tools.run module."""

from computer_use_demo.tools.run import MAX_RESPONSE_LEN, TRUNCATED_MESSAGE, maybe_truncate


class TestMaybeTruncate:
    """Tests for maybe_truncate function."""

    def test_short_content_unchanged(self):
        """Content shorter than the limit should be returned as-is."""
        content = "Hello, world!"
        assert maybe_truncate(content) == content

    def test_content_at_limit_unchanged(self):
        """Content exactly at the limit should be returned as-is."""
        content = "x" * MAX_RESPONSE_LEN
        assert maybe_truncate(content) == content

    def test_content_over_limit_truncated(self):
        """Content over the limit should be truncated with a notice appended."""
        content = "x" * (MAX_RESPONSE_LEN + 100)
        result = maybe_truncate(content)
        assert len(result) == MAX_RESPONSE_LEN + len(TRUNCATED_MESSAGE)
        assert result.startswith("x" * MAX_RESPONSE_LEN)
        assert result.endswith(TRUNCATED_MESSAGE)

    def test_custom_truncate_after(self):
        """Should respect a custom truncate_after value."""
        content = "abcdefgh"
        result = maybe_truncate(content, truncate_after=5)
        assert result == "abcde" + TRUNCATED_MESSAGE

    def test_none_truncate_after_no_truncation(self):
        """Passing None for truncate_after should disable truncation."""
        content = "x" * (MAX_RESPONSE_LEN + 1000)
        result = maybe_truncate(content, truncate_after=None)
        assert result == content

    def test_empty_string_unchanged(self):
        """Empty string should be returned as-is."""
        assert maybe_truncate("") == ""
