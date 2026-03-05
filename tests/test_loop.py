"""Tests for computer_use_demo.loop module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaToolUseBlock,
    BetaContentBlockParam,
)

from computer_use_demo.loop import (
    APIProvider,
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    normalize_provider,
    _response_to_params,
    _inject_prompt_caching,
    _maybe_filter_to_n_most_recent_images,
    _make_api_tool_result,
    _maybe_prepend_system_tool_result,
)
from computer_use_demo.tools import ToolResult


class TestAPIProvider:
    """Test the APIProvider enum."""

    def test_api_provider_values(self):
        """Test that APIProvider has expected values."""
        assert APIProvider.ANTHROPIC == "anthropic"
        assert APIProvider.KILO == "kilo"
        assert APIProvider.BEDROCK == "bedrock"
        assert APIProvider.VERTEX == "vertex"

    def test_api_provider_is_str(self):
        """Test that APIProvider members are strings."""
        assert isinstance(APIProvider.ANTHROPIC.value, str)
        assert isinstance(APIProvider.KILO.value, str)

    def test_provider_to_default_model_name(self):
        """Test that all providers have default models."""
        assert APIProvider.ANTHROPIC in PROVIDER_TO_DEFAULT_MODEL_NAME
        assert APIProvider.KILO in PROVIDER_TO_DEFAULT_MODEL_NAME
        assert APIProvider.BEDROCK in PROVIDER_TO_DEFAULT_MODEL_NAME
        assert APIProvider.VERTEX in PROVIDER_TO_DEFAULT_MODEL_NAME

    def test_default_model_names_are_strings(self):
        """Test that default model names are strings."""
        for provider, model in PROVIDER_TO_DEFAULT_MODEL_NAME.items():
            assert isinstance(model, str)
            assert len(model) > 0


class TestNormalizeProvider:
    """Test the normalize_provider function."""

    def test_normalize_provider_with_valid_enum(self):
        """Test normalizing with APIProvider enum value."""
        result = normalize_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_normalize_provider_with_valid_string(self):
        """Test normalizing with valid string values."""
        assert normalize_provider("anthropic") == APIProvider.ANTHROPIC
        assert normalize_provider("kilo") == APIProvider.KILO
        assert normalize_provider("bedrock") == APIProvider.BEDROCK
        assert normalize_provider("vertex") == APIProvider.VERTEX

    def test_normalize_provider_with_invalid_string(self):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported API provider"):
            normalize_provider("invalid_provider")

    def test_normalize_provider_with_invalid_string_empty(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported API provider"):
            normalize_provider("")

    def test_normalize_provider_preserves_type(self):
        """Test that normalized value is always APIProvider type."""
        for provider_str in ["anthropic", "kilo", "bedrock", "vertex"]:
            result = normalize_provider(provider_str)
            assert isinstance(result, APIProvider)

    def test_normalize_provider_case_sensitive(self):
        """Test that provider normalization is case-sensitive."""
        with pytest.raises(ValueError, match="Unsupported API provider"):
            normalize_provider("ANTHROPIC")
        with pytest.raises(ValueError, match="Unsupported API provider"):
            normalize_provider("Anthropic")


class TestResponseToParams:
    """Test the _response_to_params function."""

    def test_response_to_params_with_text_block(self):
        """Test converting a response with text blocks."""
        response = Mock(spec=BetaMessage)
        text_block = BetaTextBlock(type="text", text="Hello, world!")
        response.content = [text_block]

        result = _response_to_params(response)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello, world!"

    def test_response_to_params_with_multiple_blocks(self):
        """Test converting a response with multiple blocks."""
        response = Mock(spec=BetaMessage)
        text_block1 = BetaTextBlock(type="text", text="First block")
        text_block2 = BetaTextBlock(type="text", text="Second block")
        response.content = [text_block1, text_block2]

        result = _response_to_params(response)

        assert len(result) == 2
        assert result[0]["text"] == "First block"
        assert result[1]["text"] == "Second block"

    def test_response_to_params_with_tool_use_block(self):
        """Test converting a response with tool use blocks."""
        response = Mock(spec=BetaMessage)
        tool_block = Mock(spec=BetaToolUseBlock)
        tool_block.model_dump.return_value = {
            "type": "tool_use",
            "id": "tool_123",
            "name": "computer",
            "input": {"action": "screenshot"}
        }
        response.content = [tool_block]

        result = _response_to_params(response)

        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["id"] == "tool_123"

    def test_response_to_params_empty_content(self):
        """Test converting a response with no content."""
        response = Mock(spec=BetaMessage)
        response.content = []

        result = _response_to_params(response)

        assert result == []


class TestInjectPromptCaching:
    """Test the _inject_prompt_caching function."""

    def test_inject_prompt_caching_single_user_message(self):
        """Test injecting cache control for single user message."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            }
        ]

        _inject_prompt_caching(messages)

        assert "cache_control" in messages[0]["content"][-1]
        assert messages[0]["content"][-1]["cache_control"]["type"] == "ephemeral"

    def test_inject_prompt_caching_multiple_user_messages(self):
        """Test injecting cache control for multiple user messages."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Response2"}]},
            {"role": "user", "content": [{"type": "text", "text": "Third"}]},
        ]

        _inject_prompt_caching(messages)

        # Last 3 user messages should have cache control
        assert "cache_control" in messages[-1]["content"][-1]
        assert "cache_control" in messages[-3]["content"][-1]
        assert "cache_control" in messages[-5]["content"][-1]

    def test_inject_prompt_caching_removes_older_cache_controls(self):
        """Test that older cache controls are removed."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Old", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "Recent1"}]},
            {"role": "user", "content": [{"type": "text", "text": "Recent2"}]},
            {"role": "user", "content": [{"type": "text", "text": "Recent3"}]},
        ]

        _inject_prompt_caching(messages)

        # Oldest message should not have cache control
        assert "cache_control" not in messages[0]["content"][-1]
        # Recent 3 should have it
        assert "cache_control" in messages[1]["content"][-1]
        assert "cache_control" in messages[2]["content"][-1]
        assert "cache_control" in messages[3]["content"][-1]

    def test_inject_prompt_caching_skips_non_list_content(self):
        """Test that non-list content is skipped."""
        messages = [
            {"role": "user", "content": "string content"},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]

        _inject_prompt_caching(messages)

        # Only the second message should have cache control
        assert "cache_control" in messages[1]["content"][-1]

    def test_inject_prompt_caching_skips_assistant_messages(self):
        """Test that assistant messages are skipped."""
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "Assistant"}]},
            {"role": "user", "content": [{"type": "text", "text": "User"}]},
        ]

        _inject_prompt_caching(messages)

        # Only user message should have cache control
        assert "cache_control" in messages[1]["content"][-1]


class TestMaybeFilterToNMostRecentImages:
    """Test the _maybe_filter_to_n_most_recent_images function."""

    def test_filter_images_keeps_recent_images(self):
        """Test that recent images are kept."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": [
                            {"type": "image", "source": {"data": "old_image"}},
                        ]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "2",
                        "content": [
                            {"type": "image", "source": {"data": "new_image"}},
                        ]
                    }
                ]
            },
        ]

        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=1, min_removal_threshold=1)

        # First image should be removed, second kept
        assert len(messages[0]["content"][0]["content"]) == 0
        assert len(messages[1]["content"][0]["content"]) == 1
        assert messages[1]["content"][0]["content"][0]["type"] == "image"

    def test_filter_images_respects_removal_threshold(self):
        """Test that removal respects threshold."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": str(i),
                        "content": [
                            {"type": "image", "source": {"data": f"image_{i}"}},
                        ]
                    }
                ]
            }
            for i in range(15)
        ]

        # Keep 10 images with min_removal_threshold of 10
        # Total 15 images, want to keep 10, so remove 5
        # But 5 % 10 != 0, so round down to 0 removal
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=10, min_removal_threshold=10)

        # All images should still be present due to threshold
        total_images = sum(
            1 for msg in messages
            for item in msg["content"]
            if item.get("type") == "tool_result"
            for content in item.get("content", [])
            if content.get("type") == "image"
        )
        assert total_images == 15

    def test_filter_images_none_returns_messages(self):
        """Test that None images_to_keep doesn't filter."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": [
                            {"type": "image", "source": {"data": "image"}},
                        ]
                    }
                ]
            },
        ]

        original_count = len(messages[0]["content"][0]["content"])
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=None, min_removal_threshold=10)

        assert len(messages[0]["content"][0]["content"]) == original_count

    def test_filter_images_preserves_non_image_content(self):
        """Test that non-image content is preserved."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": [
                            {"type": "text", "text": "Some text"},
                            {"type": "image", "source": {"data": "image"}},
                        ]
                    }
                ]
            },
        ]

        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=0, min_removal_threshold=1)

        # Text should be preserved, image removed
        assert len(messages[0]["content"][0]["content"]) == 1
        assert messages[0]["content"][0]["content"][0]["type"] == "text"


class TestMakeApiToolResult:
    """Test the _make_api_tool_result function."""

    def test_make_api_tool_result_with_output(self):
        """Test creating tool result with output."""
        tool_result = ToolResult(output="Command executed successfully")
        result = _make_api_tool_result(tool_result, "tool_123")

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_123"
        assert result["is_error"] is False
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Command executed successfully"

    def test_make_api_tool_result_with_error(self):
        """Test creating tool result with error."""
        tool_result = ToolResult(error="Command failed")
        result = _make_api_tool_result(tool_result, "tool_456")

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_456"
        assert result["is_error"] is True
        assert result["content"] == "Command failed"

    def test_make_api_tool_result_with_base64_image(self):
        """Test creating tool result with base64 image."""
        tool_result = ToolResult(
            output="Screenshot taken",
            base64_image="abc123=="
        )
        result = _make_api_tool_result(tool_result, "tool_789")

        assert result["is_error"] is False
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["type"] == "base64"
        assert result["content"][1]["source"]["media_type"] == "image/png"
        assert result["content"][1]["source"]["data"] == "abc123=="

    def test_make_api_tool_result_with_system_message(self):
        """Test creating tool result with system message."""
        tool_result = ToolResult(
            output="Output text",
            system="System message"
        )
        result = _make_api_tool_result(tool_result, "tool_sys")

        assert result["content"][0]["text"].startswith("<system>System message</system>")
        assert "Output text" in result["content"][0]["text"]

    def test_make_api_tool_result_error_with_system(self):
        """Test creating error tool result with system message."""
        tool_result = ToolResult(
            error="Error occurred",
            system="System info"
        )
        result = _make_api_tool_result(tool_result, "tool_err")

        assert result["is_error"] is True
        assert result["content"].startswith("<system>System info</system>")
        assert "Error occurred" in result["content"]


class TestMaybePrependSystemToolResult:
    """Test the _maybe_prepend_system_tool_result function."""

    def test_prepend_system_with_system_message(self):
        """Test prepending system message when present."""
        tool_result = ToolResult(system="System info")
        result = _maybe_prepend_system_tool_result(tool_result, "Output text")

        assert result == "<system>System info</system>\nOutput text"

    def test_prepend_system_without_system_message(self):
        """Test that text is unchanged when no system message."""
        tool_result = ToolResult()
        result = _maybe_prepend_system_tool_result(tool_result, "Output text")

        assert result == "Output text"

    def test_prepend_system_empty_system(self):
        """Test with empty system message."""
        tool_result = ToolResult(system="")
        result = _maybe_prepend_system_tool_result(tool_result, "Output text")

        # Empty system should not prepend
        assert result == "Output text"

    def test_prepend_system_none_system(self):
        """Test with None system message."""
        tool_result = ToolResult(system=None)
        result = _maybe_prepend_system_tool_result(tool_result, "Output text")

        assert result == "Output text"