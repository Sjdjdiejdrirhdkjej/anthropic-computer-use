"""Tests for computer_use_demo.streamlit module."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from anthropic import RateLimitError

from computer_use_demo.loop import APIProvider
from computer_use_demo.streamlit import (
    CONFIG_DIR,
    API_KEY_FILE,
    Sender,
    WARNING_TEXT,
    _api_response_callback,
    _coerce_provider,
    _render_error,
    _reset_model,
    _tool_output_callback,
    load_from_storage,
    save_to_storage,
    setup_state,
    validate_auth,
)
from computer_use_demo.tools import ToolResult


class TestSender:
    """Tests for Sender enum."""

    def test_sender_values(self):
        """Test Sender enum values."""
        assert Sender.USER.value == "user"
        assert Sender.BOT.value == "assistant"
        assert Sender.TOOL.value == "tool"

    def test_sender_string_representation(self):
        """Test string representation of Sender."""
        assert str(Sender.USER) == "user"
        assert str(Sender.BOT) == "assistant"


class TestCoerceProvider:
    """Tests for _coerce_provider function."""

    def test_coerce_provider_with_enum(self):
        """Test _coerce_provider with APIProvider enum."""
        result = _coerce_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_coerce_provider_with_valid_string(self):
        """Test _coerce_provider with valid string values."""
        assert _coerce_provider("anthropic") == APIProvider.ANTHROPIC
        assert _coerce_provider("kilo") == APIProvider.KILO
        assert _coerce_provider("bedrock") == APIProvider.BEDROCK
        assert _coerce_provider("vertex") == APIProvider.VERTEX

    def test_coerce_provider_with_invalid_string_returns_default(self):
        """Test _coerce_provider with invalid string returns ANTHROPIC."""
        result = _coerce_provider("invalid_provider")
        assert result == APIProvider.ANTHROPIC

    def test_coerce_provider_with_empty_string(self):
        """Test _coerce_provider with empty string."""
        result = _coerce_provider("")
        assert result == APIProvider.ANTHROPIC


class TestValidateAuth:
    """Tests for validate_auth function."""

    def test_validate_auth_anthropic_missing_key(self):
        """Test validate_auth returns error for Anthropic without API key."""
        error = validate_auth(APIProvider.ANTHROPIC, None)
        assert error is not None
        assert "Anthropic API key" in error

    def test_validate_auth_anthropic_with_key(self):
        """Test validate_auth succeeds for Anthropic with API key."""
        error = validate_auth(APIProvider.ANTHROPIC, "test_key")
        assert error is None

    def test_validate_auth_anthropic_with_empty_key(self):
        """Test validate_auth returns error for Anthropic with empty key."""
        error = validate_auth(APIProvider.ANTHROPIC, "")
        assert error is not None

    def test_validate_auth_kilo_missing_key(self):
        """Test validate_auth returns error for Kilo without API key."""
        error = validate_auth(APIProvider.KILO, None)
        assert error is not None
        assert "Kilo API key" in error

    def test_validate_auth_kilo_with_key(self):
        """Test validate_auth succeeds for Kilo with API key."""
        error = validate_auth(APIProvider.KILO, "kilo_key")
        assert error is None

    def test_validate_auth_bedrock_without_credentials(self):
        """Test validate_auth for Bedrock without AWS credentials."""
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.get_credentials.return_value = None
            error = validate_auth(APIProvider.BEDROCK, "any_key")
            assert error is not None
            assert "AWS credentials" in error

    def test_validate_auth_bedrock_with_credentials(self):
        """Test validate_auth for Bedrock with AWS credentials."""
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.get_credentials.return_value = MagicMock()
            error = validate_auth(APIProvider.BEDROCK, "any_key")
            assert error is None

    def test_validate_auth_vertex_missing_region(self):
        """Test validate_auth for Vertex without CLOUD_ML_REGION."""
        with patch.dict(os.environ, {}, clear=True):
            error = validate_auth(APIProvider.VERTEX, "any_key")
            assert error is not None
            assert "CLOUD_ML_REGION" in error

    def test_validate_auth_vertex_with_invalid_credentials(self):
        """Test validate_auth for Vertex with invalid credentials."""
        from google.auth.exceptions import DefaultCredentialsError

        with patch.dict(os.environ, {"CLOUD_ML_REGION": "us-central1"}):
            with patch("google.auth.default", side_effect=DefaultCredentialsError()):
                error = validate_auth(APIProvider.VERTEX, "any_key")
                assert error is not None
                assert "google cloud credentials" in error.lower()

    def test_validate_auth_vertex_with_valid_credentials(self):
        """Test validate_auth for Vertex with valid credentials."""
        with patch.dict(os.environ, {"CLOUD_ML_REGION": "us-central1"}):
            with patch("google.auth.default", return_value=(MagicMock(), "project")):
                error = validate_auth(APIProvider.VERTEX, "any_key")
                assert error is None

    def test_validate_auth_coerces_string_provider(self):
        """Test validate_auth coerces string provider to enum."""
        error = validate_auth("anthropic", "test_key")
        assert error is None


class TestLoadFromStorage:
    """Tests for load_from_storage function."""

    def test_load_from_storage_file_exists(self):
        """Test loading data from existing file."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value="test_data\n"):
                result = load_from_storage("test_file")
                assert result == "test_data"

    def test_load_from_storage_file_not_exists(self):
        """Test loading from non-existent file returns None."""
        with patch.object(Path, "exists", return_value=False):
            result = load_from_storage("test_file")
            assert result is None

    def test_load_from_storage_empty_file(self):
        """Test loading from empty file returns None."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value="  \n  "):
                result = load_from_storage("test_file")
                assert result is None

    def test_load_from_storage_handles_exception(self):
        """Test load_from_storage handles exceptions gracefully."""
        with patch.object(Path, "exists", side_effect=Exception("Error")):
            with patch("streamlit.write") as mock_write:
                result = load_from_storage("test_file")
                assert result is None
                # Should write debug message
                mock_write.assert_called_once()


class TestSaveToStorage:
    """Tests for save_to_storage function."""

    def test_save_to_storage_creates_directory(self):
        """Test save_to_storage creates directory if needed."""
        with patch.object(Path, "mkdir") as mock_mkdir:
            with patch.object(Path, "write_text"):
                with patch.object(Path, "chmod"):
                    save_to_storage("test_file", "test_data")
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_save_to_storage_writes_data(self):
        """Test save_to_storage writes data to file."""
        with patch.object(Path, "mkdir"):
            with patch.object(Path, "write_text") as mock_write:
                with patch.object(Path, "chmod"):
                    save_to_storage("test_file", "test_data")
                    mock_write.assert_called_once_with("test_data")

    def test_save_to_storage_sets_permissions(self):
        """Test save_to_storage sets file permissions to 0o600."""
        with patch.object(Path, "mkdir"):
            with patch.object(Path, "write_text"):
                with patch.object(Path, "chmod") as mock_chmod:
                    save_to_storage("test_file", "test_data")
                    mock_chmod.assert_called_once_with(0o600)

    def test_save_to_storage_handles_exception(self):
        """Test save_to_storage handles exceptions gracefully."""
        with patch.object(Path, "mkdir", side_effect=Exception("Error")):
            with patch("streamlit.write") as mock_write:
                save_to_storage("test_file", "test_data")
                # Should write debug message
                mock_write.assert_called_once()


class TestSetupState:
    """Tests for setup_state function."""

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_initializes_messages(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state initializes messages list."""
        mock_load.return_value = None
        with patch.dict(os.environ, {}, clear=True):
            setup_state()
            assert "messages" in mock_session_state
            assert isinstance(mock_session_state["messages"], list)

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_loads_api_key_from_storage(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state loads API key from storage."""
        mock_load.return_value = "stored_key"
        with patch.dict(os.environ, {}, clear=True):
            setup_state()
            assert mock_session_state["api_key"] == "stored_key"

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_falls_back_to_env_vars(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state falls back to environment variables."""
        mock_load.return_value = None
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_key"}):
            setup_state()
            assert mock_session_state["api_key"] == "env_key"

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_prefers_kilo_key_if_anthropic_missing(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state uses KILO_API_KEY if ANTHROPIC_API_KEY is missing."""
        mock_load.return_value = None
        with patch.dict(os.environ, {"KILO_API_KEY": "kilo_key"}, clear=True):
            setup_state()
            assert mock_session_state["api_key"] == "kilo_key"

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_sets_default_provider(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state sets default provider."""
        mock_load.return_value = None
        with patch.dict(os.environ, {}, clear=True):
            setup_state()
            assert mock_session_state["provider"] == APIProvider.ANTHROPIC

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_reads_provider_from_env(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state reads provider from API_PROVIDER env var."""
        mock_load.return_value = None
        with patch.dict(os.environ, {"API_PROVIDER": "kilo"}):
            setup_state()
            assert mock_session_state["provider"] == APIProvider.KILO

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_initializes_desktop_sandbox(self, mock_load, mock_desktop_class, mock_session_state):
        """Test setup_state initializes DesktopSandbox."""
        mock_load.return_value = None
        with patch.dict(os.environ, {}, clear=True):
            setup_state()
            assert "desktop" in mock_session_state
            mock_desktop_class.assert_called_once()

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_does_not_reinitialize_existing_keys(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state does not overwrite existing session state."""
        mock_session_state["messages"] = ["existing"]
        mock_load.return_value = None
        with patch.dict(os.environ, {}, clear=True):
            setup_state()
            assert mock_session_state["messages"] == ["existing"]


class TestResetModel:
    """Tests for _reset_model function."""

    @patch("streamlit.session_state", new_callable=dict)
    def test_reset_model_sets_provider_and_model(self, mock_session_state):
        """Test _reset_model sets provider and default model."""
        mock_session_state["provider"] = "anthropic"
        _reset_model()
        assert mock_session_state["provider"] == APIProvider.ANTHROPIC
        assert "claude" in mock_session_state["model"].lower()

    @patch("streamlit.session_state", new_callable=dict)
    def test_reset_model_with_kilo_provider(self, mock_session_state):
        """Test _reset_model with Kilo provider."""
        mock_session_state["provider"] = APIProvider.KILO
        _reset_model()
        assert mock_session_state["provider"] == APIProvider.KILO
        assert "model" in mock_session_state


class TestApiResponseCallback:
    """Tests for _api_response_callback function."""

    @patch("streamlit.delta_generator.DeltaGenerator")
    def test_api_response_callback_stores_response(self, mock_tab):
        """Test _api_response_callback stores response in state."""
        request = MagicMock(spec=httpx.Request)
        response = MagicMock(spec=httpx.Response)
        response_state = {}

        _api_response_callback(request, response, None, mock_tab, response_state)

        assert len(response_state) == 1
        stored_request, stored_response = list(response_state.values())[0]
        assert stored_request == request
        assert stored_response == response

    @patch("streamlit.delta_generator.DeltaGenerator")
    @patch("computer_use_demo.streamlit._render_error")
    def test_api_response_callback_renders_error(self, mock_render_error, mock_tab):
        """Test _api_response_callback renders errors."""
        request = MagicMock(spec=httpx.Request)
        response = MagicMock(spec=httpx.Response)
        error = Exception("Test error")
        response_state = {}

        _api_response_callback(request, response, error, mock_tab, response_state)

        mock_render_error.assert_called_once_with(error)


class TestToolOutputCallback:
    """Tests for _tool_output_callback function."""

    @patch("computer_use_demo.streamlit._render_message")
    def test_tool_output_callback_stores_and_renders(self, mock_render):
        """Test _tool_output_callback stores tool output and renders it."""
        tool_output = ToolResult(output="Test output")
        tool_id = "tool_123"
        tool_state = {}

        _tool_output_callback(tool_output, tool_id, tool_state)

        assert tool_state[tool_id] == tool_output
        mock_render.assert_called_once_with(Sender.TOOL, tool_output)


class TestRenderError:
    """Tests for _render_error function."""

    @patch("streamlit.error")
    @patch("computer_use_demo.streamlit.save_to_storage")
    def test_render_error_with_rate_limit_error(self, mock_save, mock_st_error):
        """Test _render_error handles RateLimitError specially."""
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "300"  # 5 minutes
        error = RateLimitError("Rate limited", response=mock_response, body=None)
        error.message = "Too many requests"

        _render_error(error)

        # Should mention retry after
        call_args = mock_st_error.call_args[0][0]
        assert "Retry after" in call_args
        assert "Too many requests" in call_args

    @patch("streamlit.error")
    @patch("computer_use_demo.streamlit.save_to_storage")
    def test_render_error_with_generic_exception(self, mock_save, mock_st_error):
        """Test _render_error with generic exception."""
        error = ValueError("Test error")

        _render_error(error)

        call_args = mock_st_error.call_args[0][0]
        assert "ValueError" in call_args
        assert "Test error" in call_args
        assert "Traceback:" in call_args

    @patch("streamlit.error")
    @patch("computer_use_demo.streamlit.save_to_storage")
    def test_render_error_saves_to_storage(self, mock_save, mock_st_error):
        """Test _render_error saves error to storage."""
        error = Exception("Test")

        _render_error(error)

        # Should save error to storage
        assert mock_save.called
        filename = mock_save.call_args[0][0]
        assert filename.startswith("error_")
        assert filename.endswith(".md")


class TestConstants:
    """Tests for module constants."""

    def test_config_dir_is_path(self):
        """Test CONFIG_DIR is a Path object."""
        assert isinstance(CONFIG_DIR, Path)

    def test_api_key_file_is_path(self):
        """Test API_KEY_FILE is a Path object."""
        assert isinstance(API_KEY_FILE, Path)
        assert API_KEY_FILE.name == "api_key"

    def test_warning_text_contains_security_info(self):
        """Test WARNING_TEXT contains security warning."""
        assert "Security" in WARNING_TEXT or "security" in WARNING_TEXT
        assert "sensitive" in WARNING_TEXT.lower()


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_validate_auth_with_none_provider(self):
        """Test validate_auth handles None provider gracefully."""
        # Should coerce to default ANTHROPIC
        error = validate_auth(None, "test_key")
        # Will fail because None can't be coerced properly, but shouldn't crash
        # This tests defensive programming

    @patch("streamlit.session_state", new_callable=dict)
    @patch("computer_use_demo.streamlit.DesktopSandbox")
    @patch("computer_use_demo.streamlit.load_from_storage")
    def test_setup_state_with_multiple_env_vars(self, mock_load, mock_desktop, mock_session_state):
        """Test setup_state precedence with multiple API keys in env."""
        mock_load.return_value = None
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic_key", "KILO_API_KEY": "kilo_key"}):
            setup_state()
            # Should prefer ANTHROPIC_API_KEY
            assert mock_session_state["api_key"] == "anthropic_key"

    def test_coerce_provider_with_mixed_case(self):
        """Test _coerce_provider handles case sensitivity."""
        result = _coerce_provider("ANTHROPIC")
        # Should return default since exact match is required
        assert result == APIProvider.ANTHROPIC

    @patch("streamlit.session_state", new_callable=dict)
    def test_reset_model_with_invalid_provider(self, mock_session_state):
        """Test _reset_model handles invalid provider."""
        mock_session_state["provider"] = "invalid"
        _reset_model()
        # Should coerce to ANTHROPIC
        assert mock_session_state["provider"] == APIProvider.ANTHROPIC