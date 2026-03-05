"""Tests for computer_use_demo.loop module."""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from anthropic import APIError, APIResponseValidationError, APIStatusError
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaToolResultBlockParam,
    BetaToolUseBlock,
)

from computer_use_demo.loop import (
    COMPUTER_USE_BETA_FLAG,
    PROMPT_CACHING_BETA_FLAG,
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    SYSTEM_PROMPT,
    APIProvider,
    _inject_prompt_caching,
    _make_api_tool_result,
    _maybe_filter_to_n_most_recent_images,
    _maybe_prepend_system_tool_result,
    _response_to_params,
    normalize_provider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult


class TestAPIProvider:
    """Tests for APIProvider enum."""

    def test_provider_values(self):
        """Test that APIProvider has expected values."""
        assert APIProvider.ANTHROPIC.value == "anthropic"
        assert APIProvider.KILO.value == "kilo"
        assert APIProvider.BEDROCK.value == "bedrock"
        assert APIProvider.VERTEX.value == "vertex"

    def test_provider_string_representation(self):
        """Test string representation of APIProvider members."""
        assert str(APIProvider.ANTHROPIC) == "anthropic"
        assert str(APIProvider.KILO) == "kilo"

    def test_provider_equality(self):
        """Test APIProvider equality comparisons."""
        assert APIProvider.ANTHROPIC == "anthropic"
        assert APIProvider.KILO == "kilo"


class TestNormalizeProvider:
    """Tests for normalize_provider function."""

    def test_normalize_provider_with_enum(self):
        """Test normalize_provider with APIProvider enum."""
        result = normalize_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_normalize_provider_with_valid_string(self):
        """Test normalize_provider with valid string values."""
        assert normalize_provider("anthropic") == APIProvider.ANTHROPIC
        assert normalize_provider("kilo") == APIProvider.KILO
        assert normalize_provider("bedrock") == APIProvider.BEDROCK
        assert normalize_provider("vertex") == APIProvider.VERTEX

    def test_normalize_provider_with_invalid_string(self):
        """Test normalize_provider with invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported API provider"):
            normalize_provider("invalid_provider")

    def test_normalize_provider_error_message(self):
        """Test that normalize_provider error includes the invalid provider name."""
        try:
            normalize_provider("unknown")
        except ValueError as e:
            assert "unknown" in str(e)
            assert "Unsupported API provider" in str(e)


class TestProviderToDefaultModelName:
    """Tests for PROVIDER_TO_DEFAULT_MODEL_NAME mapping."""

    def test_all_providers_have_default_models(self):
        """Test that all providers have default model mappings."""
        for provider in APIProvider:
            assert provider in PROVIDER_TO_DEFAULT_MODEL_NAME
            assert isinstance(PROVIDER_TO_DEFAULT_MODEL_NAME[provider], str)
            assert len(PROVIDER_TO_DEFAULT_MODEL_NAME[provider]) > 0

    def test_anthropic_default_model(self):
        """Test Anthropic provider default model."""
        assert "claude" in PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC].lower()

    def test_kilo_default_model(self):
        """Test Kilo provider default model."""
        assert "claude" in PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.KILO].lower()


class TestMaybePrependSystemToolResult:
    """Tests for _maybe_prepend_system_tool_result function."""

    def test_prepend_with_system_message(self):
        """Test prepending system message to result text."""
        result = ToolResult(system="System info")
        text = "Output text"
        output = _maybe_prepend_system_tool_result(result, text)
        assert output == "<system>System info</system>\nOutput text"

    def test_no_prepend_without_system_message(self):
        """Test that text is unchanged without system message."""
        result = ToolResult()
        text = "Output text"
        output = _maybe_prepend_system_tool_result(result, text)
        assert output == "Output text"

    def test_prepend_with_empty_text(self):
        """Test prepending to empty text."""
        result = ToolResult(system="System info")
        text = ""
        output = _maybe_prepend_system_tool_result(result, text)
        assert output == "<system>System info</system>\n"


class TestMakeApiToolResult:
    """Tests for _make_api_tool_result function."""

    def test_make_tool_result_with_error(self):
        """Test creating tool result with error."""
        result = ToolResult(error="Error message")
        tool_use_id = "test_id"
        api_result = _make_api_tool_result(result, tool_use_id)

        assert api_result["type"] == "tool_result"
        assert api_result["tool_use_id"] == "test_id"
        assert api_result["is_error"] is True
        assert api_result["content"] == "Error message"

    def test_make_tool_result_with_output(self):
        """Test creating tool result with output."""
        result = ToolResult(output="Success output")
        tool_use_id = "test_id"
        api_result = _make_api_tool_result(result, tool_use_id)

        assert api_result["type"] == "tool_result"
        assert api_result["tool_use_id"] == "test_id"
        assert api_result["is_error"] is False
        assert isinstance(api_result["content"], list)
        assert len(api_result["content"]) == 1
        assert api_result["content"][0]["type"] == "text"
        assert api_result["content"][0]["text"] == "Success output"

    def test_make_tool_result_with_image(self):
        """Test creating tool result with base64 image."""
        result = ToolResult(output="Output", base64_image="base64data")
        tool_use_id = "test_id"
        api_result = _make_api_tool_result(result, tool_use_id)

        assert len(api_result["content"]) == 2
        assert api_result["content"][0]["type"] == "text"
        assert api_result["content"][1]["type"] == "image"
        assert api_result["content"][1]["source"]["type"] == "base64"
        assert api_result["content"][1]["source"]["data"] == "base64data"

    def test_make_tool_result_with_system_message(self):
        """Test creating tool result with system message in output."""
        result = ToolResult(output="Output", system="System")
        tool_use_id = "test_id"
        api_result = _make_api_tool_result(result, tool_use_id)

        assert "<system>System</system>" in api_result["content"][0]["text"]


class TestResponseToParams:
    """Tests for _response_to_params function."""

    def test_response_with_text_block(self):
        """Test converting response with text block."""
        text_block = BetaTextBlock(type="text", text="Test response")
        response = MagicMock(spec=BetaMessage)
        response.content = [text_block]

        params = _response_to_params(response)

        assert len(params) == 1
        assert params[0]["type"] == "text"
        assert params[0]["text"] == "Test response"

    def test_response_with_tool_use_block(self):
        """Test converting response with tool use block."""
        tool_block = MagicMock(spec=BetaToolUseBlock)
        tool_block.model_dump.return_value = {
            "type": "tool_use",
            "id": "tool_id",
            "name": "computer",
            "input": {},
        }
        response = MagicMock(spec=BetaMessage)
        response.content = [tool_block]

        params = _response_to_params(response)

        assert len(params) == 1
        assert params[0]["type"] == "tool_use"

    def test_response_with_mixed_blocks(self):
        """Test converting response with both text and tool blocks."""
        text_block = BetaTextBlock(type="text", text="Text")
        tool_block = MagicMock(spec=BetaToolUseBlock)
        tool_block.model_dump.return_value = {"type": "tool_use"}

        response = MagicMock(spec=BetaMessage)
        response.content = [text_block, tool_block]

        params = _response_to_params(response)

        assert len(params) == 2


class TestInjectPromptCaching:
    """Tests for _inject_prompt_caching function."""

    def test_inject_caching_to_recent_messages(self):
        """Test that cache control is added to the 3 most recent user messages."""
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

        # Check that the last 3 user messages have cache_control
        user_messages = [m for m in messages if m["role"] == "user"]
        assert "cache_control" in user_messages[-1]["content"][-1]
        assert "cache_control" in user_messages[-2]["content"][-1]
        assert "cache_control" in user_messages[-3]["content"][-1]

    def test_inject_caching_removes_from_older_messages(self):
        """Test that cache_control is removed from messages older than the 3 most recent."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "msg1", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "msg2"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg3"}]},
            {"role": "user", "content": [{"type": "text", "text": "msg4"}]},
        ]

        _inject_prompt_caching(messages)

        # First message should have cache_control removed
        assert "cache_control" not in messages[0]["content"][-1]
        # Last 3 should have it
        assert "cache_control" in messages[1]["content"][-1]
        assert "cache_control" in messages[2]["content"][-1]
        assert "cache_control" in messages[3]["content"][-1]

    def test_inject_caching_with_non_list_content(self):
        """Test that inject_prompt_caching handles non-list content gracefully."""
        messages = [
            {"role": "user", "content": "string content"},
            {"role": "user", "content": [{"type": "text", "text": "list content"}]},
        ]

        # Should not raise an error
        _inject_prompt_caching(messages)


class TestMaybeFilterToNMostRecentImages:
    """Tests for _maybe_filter_to_n_most_recent_images function."""

    def test_filter_keeps_recent_images(self):
        """Test that filtering keeps only the N most recent images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "result1"},
                            {"type": "image", "source": "image1"},
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
                            {"type": "image", "source": "image2"},
                            {"type": "image", "source": "image3"},
                        ],
                    }
                ],
            },
        ]

        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=1, min_removal_threshold=1)

        # Count remaining images
        image_count = 0
        for msg in messages:
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        for content in block.get("content", []):
                            if isinstance(content, dict) and content.get("type") == "image":
                                image_count += 1

        assert image_count == 1

    def test_filter_with_none_returns_early(self):
        """Test that passing None for images_to_keep does nothing."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": [{"type": "image", "source": "image1"}]}
                ],
            }
        ]

        original = messages.copy()
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=None, min_removal_threshold=10)

        # Messages should be unchanged
        assert messages == original

    def test_filter_preserves_text_content(self):
        """Test that filtering preserves non-image content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "important text"},
                            {"type": "image", "source": "image1"},
                        ],
                    }
                ],
            }
        ]

        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=0, min_removal_threshold=1)

        # Text should remain, image should be removed
        tool_result = messages[0]["content"][0]
        assert any(c.get("type") == "text" for c in tool_result.get("content", []))
        # Verify image was removed
        image_count = sum(1 for c in tool_result.get("content", []) if c.get("type") == "image")
        assert image_count == 0

    def test_filter_respects_min_removal_threshold(self):
        """Test that filtering respects the minimum removal threshold."""
        messages = []
        # Create 15 images
        for i in range(15):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": [{"type": "image", "source": f"image{i}"}]}
                    ],
                }
            )

        # Keep 10 images with threshold of 10
        # Should remove 5 images, but threshold would reduce to 0 (5 % 10 = 5, 5 - 5 = 0)
        _maybe_filter_to_n_most_recent_images(messages, images_to_keep=10, min_removal_threshold=10)

        # Count images - should still have all 15 because removal amount didn't meet threshold
        image_count = sum(
            1
            for msg in messages
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
            for content in block.get("content", [])
            if isinstance(content, dict) and content.get("type") == "image"
        )
        assert image_count == 15


@pytest.mark.asyncio
class TestSamplingLoop:
    """Tests for sampling_loop function."""

    @pytest.fixture
    def mock_desktop(self):
        """Create a mock DesktopSandbox."""
        return MagicMock()

    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callback functions."""
        return {
            "output_callback": Mock(),
            "tool_output_callback": Mock(),
            "api_response_callback": Mock(),
        }

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_sampling_loop_with_text_response(self, mock_desktop, mock_callbacks):
        """Test sampling loop with a simple text response (no tools)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        # Mock the API client response
        mock_response = MagicMock(spec=BetaMessage)
        mock_response.content = [BetaTextBlock(type="text", text="Hi there")]

        mock_raw_response = MagicMock()
        mock_raw_response.parse.return_value = mock_response
        mock_raw_response.http_response.request = MagicMock()
        mock_raw_response.http_response = MagicMock()

        with patch("computer_use_demo.loop.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.beta.messages.with_raw_response.create.return_value = mock_raw_response
            mock_anthropic_class.return_value = mock_client

            result = await sampling_loop(
                desktop=mock_desktop,
                model="claude-3-5-sonnet-20241022",
                provider=APIProvider.ANTHROPIC,
                system_prompt_suffix="",
                messages=messages,
                api_key="test_key",
                **mock_callbacks,
            )

            # Should have added assistant response
            assert len(result) == 2
            assert result[1]["role"] == "assistant"

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_sampling_loop_handles_api_status_error(self, mock_desktop, mock_callbacks):
        """Test that sampling loop handles APIStatusError properly."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        mock_request = MagicMock()
        mock_response = MagicMock()
        error = APIStatusError("Error", response=mock_response, body=None)
        error.request = mock_request

        with patch("computer_use_demo.loop.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.beta.messages.with_raw_response.create.side_effect = error
            mock_anthropic_class.return_value = mock_client

            result = await sampling_loop(
                desktop=mock_desktop,
                model="claude-3-5-sonnet-20241022",
                provider=APIProvider.ANTHROPIC,
                system_prompt_suffix="",
                messages=messages,
                api_key="test_key",
                **mock_callbacks,
            )

            # Should call error callback
            mock_callbacks["api_response_callback"].assert_called_once()
            # Should return original messages
            assert result == messages

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_sampling_loop_handles_api_error(self, mock_desktop, mock_callbacks):
        """Test that sampling loop handles APIError properly."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        mock_request = MagicMock()
        # APIError requires request as first positional argument
        error = APIError("Error", request=mock_request, body={"error": "test"})

        with patch("computer_use_demo.loop.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.beta.messages.with_raw_response.create.side_effect = error
            mock_anthropic_class.return_value = mock_client

            result = await sampling_loop(
                desktop=mock_desktop,
                model="claude-3-5-sonnet-20241022",
                provider=APIProvider.ANTHROPIC,
                system_prompt_suffix="",
                messages=messages,
                api_key="test_key",
                **mock_callbacks,
            )

            # Should return original messages
            assert result == messages

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_sampling_loop_uses_kilo_provider(self, mock_desktop, mock_callbacks):
        """Test sampling loop with Kilo provider."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        mock_response = MagicMock(spec=BetaMessage)
        mock_response.content = [BetaTextBlock(type="text", text="Response")]

        mock_raw_response = MagicMock()
        mock_raw_response.parse.return_value = mock_response
        mock_raw_response.http_response.request = MagicMock()
        mock_raw_response.http_response = MagicMock()

        with patch("computer_use_demo.loop.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.beta.messages.with_raw_response.create.return_value = mock_raw_response
            mock_anthropic_class.return_value = mock_client

            # Set KILO_API_BASE_URL env var
            with patch.dict(os.environ, {"KILO_API_BASE_URL": "https://custom.kilo.ai/anthropic", "WIDTH": "1920", "HEIGHT": "1080"}):
                await sampling_loop(
                    desktop=mock_desktop,
                    model="claude-3-5-sonnet-20241022",
                    provider=APIProvider.KILO,
                    system_prompt_suffix="",
                    messages=messages,
                    api_key="kilo_key",
                    **mock_callbacks,
                )

            # Verify Anthropic was called with custom base_url
            call_kwargs = mock_anthropic_class.call_args[1]
            assert call_kwargs["api_key"] == "kilo_key"
            assert call_kwargs["base_url"] == "https://custom.kilo.ai/anthropic"

    @patch.dict(os.environ, {"WIDTH": "1920", "HEIGHT": "1080"})
    async def test_sampling_loop_with_system_prompt_suffix(self, mock_desktop, mock_callbacks):
        """Test that system prompt suffix is appended."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        mock_response = MagicMock(spec=BetaMessage)
        mock_response.content = [BetaTextBlock(type="text", text="Response")]

        mock_raw_response = MagicMock()
        mock_raw_response.parse.return_value = mock_response
        mock_raw_response.http_response.request = MagicMock()
        mock_raw_response.http_response = MagicMock()

        with patch("computer_use_demo.loop.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.beta.messages.with_raw_response.create.return_value = mock_raw_response
            mock_anthropic_class.return_value = mock_client

            await sampling_loop(
                desktop=mock_desktop,
                model="claude-3-5-sonnet-20241022",
                provider=APIProvider.ANTHROPIC,
                system_prompt_suffix="Additional instructions",
                messages=messages,
                api_key="test_key",
                **mock_callbacks,
            )

            # Check that system prompt includes suffix
            create_call = mock_client.beta.messages.with_raw_response.create.call_args
            system_param = create_call[1]["system"][0]
            assert "Additional instructions" in system_param["text"]


class TestSystemPrompt:
    """Tests for SYSTEM_PROMPT constant."""

    def test_system_prompt_contains_key_info(self):
        """Test that system prompt contains essential information."""
        assert "Ubuntu" in SYSTEM_PROMPT
        assert "bash" in SYSTEM_PROMPT
        assert "firefox" in SYSTEM_PROMPT or "Firefox" in SYSTEM_PROMPT

    def test_system_prompt_includes_date(self):
        """Test that system prompt includes current date."""
        from datetime import datetime

        date_str = datetime.today().strftime("%A, %B")
        # Should contain at least month name
        assert any(month in SYSTEM_PROMPT for month in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])


class TestBetaFlags:
    """Tests for beta flag constants."""

    def test_computer_use_beta_flag(self):
        """Test COMPUTER_USE_BETA_FLAG value."""
        assert COMPUTER_USE_BETA_FLAG == "computer-use-2024-10-22"

    def test_prompt_caching_beta_flag(self):
        """Test PROMPT_CACHING_BETA_FLAG value."""
        assert PROMPT_CACHING_BETA_FLAG == "prompt-caching-2024-07-31"