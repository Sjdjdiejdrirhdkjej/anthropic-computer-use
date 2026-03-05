"""Tests for computer_use_demo.loop module."""

import pytest
from unittest.mock import MagicMock, Mock

from anthropic.types.beta import (
    BetaMessage,
    BetaTextBlock,
    BetaToolUseBlock,
)

from computer_use_demo.loop import (
    APIProvider,
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    normalize_provider,
    _maybe_filter_to_n_most_recent_images,
    _response_to_params,
    _inject_prompt_caching,
    _make_api_tool_result,
    _maybe_prepend_system_tool_result,
)
from computer_use_demo.tools import ToolResult


class TestAPIProvider:
    """Tests for APIProvider enum."""

    def test_api_provider_values(self):
        """Test that APIProvider has the expected values."""
        assert APIProvider.ANTHROPIC == "anthropic"
        assert APIProvider.KILO == "kilo"
        assert APIProvider.BEDROCK == "bedrock"
        assert APIProvider.VERTEX == "vertex"

    def test_api_provider_string_behavior(self):
        """Test that APIProvider members behave like strings."""
        assert str(APIProvider.ANTHROPIC) == "anthropic"
        assert APIProvider.ANTHROPIC == "anthropic"


class TestNormalizeProvider:
    """Tests for normalize_provider function."""

    def test_normalize_provider_with_enum(self):
        """Test normalize_provider with APIProvider enum."""
        result = normalize_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_normalize_provider_with_valid_string(self):
        """Test normalize_provider with valid string."""
        result = normalize_provider("anthropic")
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

        result = normalize_provider("kilo")
        assert result == APIProvider.KILO

        result = normalize_provider("bedrock")
        assert result == APIProvider.BEDROCK

        result = normalize_provider("vertex")
        assert result == APIProvider.VERTEX

    def test_normalize_provider_with_invalid_string(self):
        """Test normalize_provider with invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported API provider: invalid"):
            normalize_provider("invalid")

        with pytest.raises(ValueError, match="Unsupported API provider: ANTHROPIC"):
            normalize_provider("ANTHROPIC")  # case-sensitive

    def test_normalize_provider_preserves_enum_type(self):
        """Test that normalize_provider preserves enum type."""
        for provider in APIProvider:
            result = normalize_provider(provider)
            assert result is provider


class TestProviderToDefaultModel:
    """Tests for PROVIDER_TO_DEFAULT_MODEL_NAME mapping."""

    def test_all_providers_have_default_models(self):
        """Test that all providers have a default model."""
        for provider in APIProvider:
            assert provider in PROVIDER_TO_DEFAULT_MODEL_NAME
            assert isinstance(PROVIDER_TO_DEFAULT_MODEL_NAME[provider], str)
            assert len(PROVIDER_TO_DEFAULT_MODEL_NAME[provider]) > 0


class TestMaybeFilterToNMostRecentImages:
    """Tests for _maybe_filter_to_n_most_recent_images function."""

    def test_filter_with_none_images_to_keep(self):
        """Test that no filtering occurs when images_to_keep is None."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "image", "source": {}}],
                    }
                ],
            }
        ]
        original_messages = messages.copy()
        _maybe_filter_to_n_most_recent_images(messages, None, 10)
        assert messages == original_messages

    def test_filter_removes_old_images(self):
        """Test that old images are removed."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "image", "source": {}},
                            {"type": "text", "text": "result1"},
                        ],
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "image", "source": {}},
                            {"type": "text", "text": "result2"},
                        ],
                    }
                ],
            },
        ]
        _maybe_filter_to_n_most_recent_images(messages, 1, 1)

        # First image should be removed, second kept
        assert len(messages[0]["content"][0]["content"]) == 1
        assert messages[0]["content"][0]["content"][0]["type"] == "text"

        assert len(messages[1]["content"][0]["content"]) == 2
        assert any(c["type"] == "image" for c in messages[1]["content"][0]["content"])

    def test_filter_respects_min_removal_threshold(self):
        """Test that images are removed in chunks based on min_removal_threshold."""
        messages = []
        for i in range(15):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "image", "source": {}}],
                        }
                    ],
                }
            )

        # Keep 10 images, min_removal_threshold=10
        # Total images: 15, to remove: 5, but 5 % 10 = 5, so remove 0
        _maybe_filter_to_n_most_recent_images(messages, 10, 10)

        remaining_images = sum(
            1
            for msg in messages
            for item in msg.get("content", [])
            if isinstance(item, dict) and item.get("type") == "tool_result"
            for content in item.get("content", [])
            if isinstance(content, dict) and content.get("type") == "image"
        )
        assert remaining_images == 15  # No images removed due to threshold

    def test_filter_preserves_non_image_content(self):
        """Test that non-image content is preserved."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "some text"},
                            {"type": "image", "source": {}},
                        ],
                    }
                ],
            }
        ]
        _maybe_filter_to_n_most_recent_images(messages, 0, 1)

        # Text should be preserved, image removed
        assert len(messages[0]["content"][0]["content"]) == 1
        assert messages[0]["content"][0]["content"][0]["type"] == "text"


class TestResponseToParams:
    """Tests for _response_to_params function."""

    def test_response_to_params_with_text_block(self):
        """Test converting BetaMessage with text block to params."""
        mock_text_block = MagicMock(spec=BetaTextBlock)
        mock_text_block.text = "Hello, world!"

        mock_response = MagicMock(spec=BetaMessage)
        mock_response.content = [mock_text_block]

        result = _response_to_params(mock_response)

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello, world!"

    def test_response_to_params_with_tool_use_block(self):
        """Test converting BetaMessage with tool use block to params."""
        mock_tool_block = MagicMock(spec=BetaToolUseBlock)
        mock_tool_block.model_dump.return_value = {
            "type": "tool_use",
            "id": "tool_123",
            "name": "computer",
            "input": {"action": "screenshot"},
        }

        mock_response = MagicMock(spec=BetaMessage)
        mock_response.content = [mock_tool_block]

        result = _response_to_params(mock_response)

        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["id"] == "tool_123"

    def test_response_to_params_with_mixed_content(self):
        """Test converting BetaMessage with mixed text and tool use blocks."""
        mock_text_block = MagicMock(spec=BetaTextBlock)
        mock_text_block.text = "Let me take a screenshot"

        mock_tool_block = MagicMock(spec=BetaToolUseBlock)
        mock_tool_block.model_dump.return_value = {
            "type": "tool_use",
            "id": "tool_456",
            "name": "computer",
        }

        mock_response = MagicMock(spec=BetaMessage)
        mock_response.content = [mock_text_block, mock_tool_block]

        result = _response_to_params(mock_response)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"


class TestInjectPromptCaching:
    """Tests for _inject_prompt_caching function."""

    def test_inject_prompt_caching_on_recent_messages(self):
        """Test that cache control is added to 3 most recent user messages."""
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

        # Check that the last 3 user messages have cache control
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assert "cache_control" in user_messages[-1]["content"][-1]
        assert "cache_control" in user_messages[-2]["content"][-1]
        assert "cache_control" in user_messages[-3]["content"][-1]

        # First user message should not have cache control
        if len(user_messages) > 3:
            assert "cache_control" not in user_messages[0]["content"][-1]

    def test_inject_prompt_caching_removes_old_cache_control(self):
        """Test that old cache control is removed from the 4th most recent message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg1", "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "msg2"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg3"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg4"}]},
        ]

        _inject_prompt_caching(messages)

        # Fourth-most-recent message (msg1) should have cache control removed
        assert "cache_control" not in messages[0]["content"][-1]
        # Most recent 3 messages should have cache control
        assert "cache_control" in messages[1]["content"][-1]
        assert "cache_control" in messages[2]["content"][-1]
        assert "cache_control" in messages[3]["content"][-1]


class TestMakeApiToolResult:
    """Tests for _make_api_tool_result function."""

    def test_make_api_tool_result_with_output(self):
        """Test creating tool result with output."""
        result = ToolResult(output="Command executed successfully")

        api_result = _make_api_tool_result(result, "tool_123")

        assert api_result["type"] == "tool_result"
        assert api_result["tool_use_id"] == "tool_123"
        assert api_result["is_error"] is False
        assert len(api_result["content"]) == 1
        assert api_result["content"][0]["type"] == "text"
        assert api_result["content"][0]["text"] == "Command executed successfully"

    def test_make_api_tool_result_with_error(self):
        """Test creating tool result with error."""
        result = ToolResult(error="Command failed")

        api_result = _make_api_tool_result(result, "tool_456")

        assert api_result["type"] == "tool_result"
        assert api_result["tool_use_id"] == "tool_456"
        assert api_result["is_error"] is True
        assert api_result["content"] == "Command failed"

    def test_make_api_tool_result_with_base64_image(self):
        """Test creating tool result with base64 image."""
        result = ToolResult(output="Screenshot taken", base64_image="base64data")

        api_result = _make_api_tool_result(result, "tool_789")

        assert len(api_result["content"]) == 2
        assert api_result["content"][0]["type"] == "text"
        assert api_result["content"][1]["type"] == "image"
        assert api_result["content"][1]["source"]["type"] == "base64"
        assert api_result["content"][1]["source"]["data"] == "base64data"

    def test_make_api_tool_result_with_system_info(self):
        """Test creating tool result with system info."""
        result = ToolResult(output="Output", system="System info")

        api_result = _make_api_tool_result(result, "tool_999")

        assert api_result["content"][0]["text"] == "<system>System info</system>\nOutput"


class TestMaybePrependSystemToolResult:
    """Tests for _maybe_prepend_system_tool_result function."""

    def test_prepend_system_info_when_present(self):
        """Test that system info is prepended when present."""
        result = ToolResult(system="System message")

        output = _maybe_prepend_system_tool_result(result, "Result text")

        assert output == "<system>System message</system>\nResult text"

    def test_no_prepend_when_system_is_none(self):
        """Test that nothing is prepended when system is None."""
        result = ToolResult()

        output = _maybe_prepend_system_tool_result(result, "Result text")

        assert output == "Result text"

    def test_prepend_with_empty_result_text(self):
        """Test prepending system info with empty result text."""
        result = ToolResult(system="System only")

        output = _maybe_prepend_system_tool_result(result, "")

        assert output == "<system>System only</system>\n"