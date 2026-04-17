"""Tests for computer_use_demo.loop utility functions."""

import pytest

from computer_use_demo.compat import StrEnum
from computer_use_demo.loop import (
    APIProvider,
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    _inject_prompt_caching,
    _make_api_tool_result,
    _maybe_filter_to_n_most_recent_images,
    _maybe_prepend_system_tool_result,
    normalize_provider,
)
from computer_use_demo.tools import ToolResult


class TestAPIProvider:
    """Tests for APIProvider enum."""

    def test_provider_values(self):
        """APIProvider should have the expected string values."""
        assert APIProvider.ANTHROPIC.value == "anthropic"
        assert APIProvider.KILO.value == "kilo"
        assert APIProvider.BEDROCK.value == "bedrock"
        assert APIProvider.VERTEX.value == "vertex"

    def test_provider_is_strenum(self):
        """APIProvider should be a StrEnum subclass."""
        assert issubclass(APIProvider, StrEnum)

    def test_provider_string_comparison(self):
        """APIProvider members should compare equal to their string values."""
        assert APIProvider.ANTHROPIC == "anthropic"


class TestNormalizeProvider:
    """Tests for normalize_provider function."""

    def test_normalize_with_enum(self):
        """normalize_provider should return the same enum when given one."""
        result = normalize_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_normalize_with_valid_string(self):
        """normalize_provider should convert valid strings to APIProvider."""
        assert normalize_provider("anthropic") == APIProvider.ANTHROPIC
        assert normalize_provider("kilo") == APIProvider.KILO
        assert normalize_provider("bedrock") == APIProvider.BEDROCK
        assert normalize_provider("vertex") == APIProvider.VERTEX

    def test_normalize_with_invalid_string(self):
        """normalize_provider should raise ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Unsupported API provider"):
            normalize_provider("invalid")

    def test_normalize_error_includes_provider_name(self):
        """normalize_provider error message should include the invalid name."""
        with pytest.raises(ValueError, match="unknown"):
            normalize_provider("unknown")


class TestProviderToDefaultModelName:
    """Tests for PROVIDER_TO_DEFAULT_MODEL_NAME mapping."""

    def test_all_providers_have_defaults(self):
        """Every APIProvider member should have a default model mapping."""
        for provider in APIProvider:
            assert provider in PROVIDER_TO_DEFAULT_MODEL_NAME
            assert isinstance(PROVIDER_TO_DEFAULT_MODEL_NAME[provider], str)
            assert len(PROVIDER_TO_DEFAULT_MODEL_NAME[provider]) > 0

    def test_anthropic_default_contains_claude(self):
        """Anthropic default model name should contain 'claude'."""
        assert "claude" in PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC].lower()


class TestMaybePrependSystemToolResult:
    """Tests for _maybe_prepend_system_tool_result function."""

    def test_prepend_with_system_message(self):
        """Should prepend system message wrapped in <system> tags."""
        result = ToolResult(system="System info")
        output = _maybe_prepend_system_tool_result(result, "Output text")
        assert output == "<system>System info</system>\nOutput text"

    def test_no_prepend_without_system(self):
        """Should return text unchanged when no system message."""
        result = ToolResult()
        output = _maybe_prepend_system_tool_result(result, "Output text")
        assert output == "Output text"

    def test_prepend_with_empty_text(self):
        """Should prepend system message even when text is empty."""
        result = ToolResult(system="System info")
        output = _maybe_prepend_system_tool_result(result, "")
        assert output == "<system>System info</system>\n"


class TestMakeApiToolResult:
    """Tests for _make_api_tool_result function."""

    def test_error_result(self):
        """Should create error tool result with is_error=True."""
        result = ToolResult(error="Error message")
        api_result = _make_api_tool_result(result, "tool_123")
        assert api_result["type"] == "tool_result"
        assert api_result["tool_use_id"] == "tool_123"
        assert api_result["is_error"] is True
        assert api_result["content"] == "Error message"

    def test_output_result(self):
        """Should create tool result with text content."""
        result = ToolResult(output="Success output")
        api_result = _make_api_tool_result(result, "tool_123")
        assert api_result["is_error"] is False
        assert isinstance(api_result["content"], list)
        assert api_result["content"][0]["type"] == "text"
        assert api_result["content"][0]["text"] == "Success output"

    def test_output_with_image(self):
        """Should include image block when base64_image is present."""
        result = ToolResult(output="Output", base64_image="base64data")
        api_result = _make_api_tool_result(result, "tool_123")
        assert len(api_result["content"]) == 2
        assert api_result["content"][1]["type"] == "image"
        assert api_result["content"][1]["source"]["data"] == "base64data"

    def test_output_with_system_message(self):
        """Should prepend system message to output text."""
        result = ToolResult(output="Output", system="System")
        api_result = _make_api_tool_result(result, "tool_123")
        assert "<system>System</system>" in api_result["content"][0]["text"]


class TestInjectPromptCaching:
    """Tests for _inject_prompt_caching function."""

    def test_inject_caching_to_recent_user_messages(self):
        """Should add cache_control to the 3 most recent user messages."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "msg1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "resp1"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "resp2"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg3"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "resp3"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg4"}]},
        ]
        _inject_prompt_caching(messages)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert "cache_control" in user_msgs[-1]["content"][-1]
        assert "cache_control" in user_msgs[-2]["content"][-1]
        assert "cache_control" in user_msgs[-3]["content"][-1]

    def test_remove_caching_from_older_messages(self):
        """Should remove cache_control from messages beyond the 3 most recent."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "msg1", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "msg2"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg3"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg4"}]},
        ]
        _inject_prompt_caching(messages)
        assert "cache_control" not in messages[0]["content"][-1]
        assert "cache_control" in messages[1]["content"][-1]
        assert "cache_control" in messages[2]["content"][-1]
        assert "cache_control" in messages[3]["content"][-1]


class TestMaybeFilterToNMostRecentImages:
    """Tests for _maybe_filter_to_n_most_recent_images function."""

    def test_none_images_to_keep_returns_early(self):
        """Should return messages unchanged when images_to_keep is None."""
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image", "source": "img"}]}]}
        ]
        original = str(messages)
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=None, min_removal_threshold=10)
        assert str(messages) == original

    def test_filter_removes_oldest_images(self):
        """Should remove oldest images, keeping only the N most recent."""
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "text", "text": "t1"}, {"type": "image", "source": "img1"}]}]},
            {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image", "source": "img2"}, {"type": "image", "source": "img3"}]}]},
        ]
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=1, min_removal_threshold=1)
        image_count = sum(
            1
            for msg in messages
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
            for content in block.get("content", [])
            if isinstance(content, dict) and content.get("type") == "image"
        )
        assert image_count == 1

    def test_filter_preserves_text_content(self):
        """Should preserve non-image content when filtering."""
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "text", "text": "important"}, {"type": "image", "source": "img"}]}]},
        ]
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=0, min_removal_threshold=1)
        tool_result = messages[0]["content"][0]
        text_items = [c for c in tool_result.get("content", []) if c.get("type") == "text"]
        image_items = [c for c in tool_result.get("content", []) if c.get("type") == "image"]
        assert len(text_items) == 1
        assert len(image_items) == 0

    def test_filter_respects_min_removal_threshold(self):
        """Should not remove images if the removal count is below threshold."""
        messages = []
        for i in range(15):
            messages.append(
                {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image", "source": f"img{i}"}]}]}
            )
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=10, min_removal_threshold=10)
        image_count = sum(
            1
            for msg in messages
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
            for content in block.get("content", [])
            if isinstance(content, dict) and content.get("type") == "image"
        )
        # 15 images, keep 10 → remove 5, but threshold=10 → 5%10=5, 5-5=0 removed
        assert image_count == 15
