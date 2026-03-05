"""Tests for computer_use_demo.streamlit module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from computer_use_demo.streamlit import (
    Sender,
    _coerce_provider,
    validate_auth,
    load_from_storage,
    save_to_storage,
)
from computer_use_demo.loop import APIProvider


class TestSender:
    """Test the Sender enum."""

    def test_sender_values(self):
        """Test that Sender has expected values."""
        assert Sender.USER == "user"
        assert Sender.BOT == "assistant"
        assert Sender.TOOL == "tool"

    def test_sender_is_str(self):
        """Test that Sender members are strings."""
        assert isinstance(Sender.USER.value, str)
        assert isinstance(Sender.BOT.value, str)
        assert isinstance(Sender.TOOL.value, str)


class TestCoerceProvider:
    """Test the _coerce_provider function."""

    def test_coerce_provider_with_valid_enum(self):
        """Test coercing with APIProvider enum value."""
        result = _coerce_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_coerce_provider_with_valid_string(self):
        """Test coercing with valid string values."""
        assert _coerce_provider("anthropic") == APIProvider.ANTHROPIC
        assert _coerce_provider("kilo") == APIProvider.KILO
        assert _coerce_provider("bedrock") == APIProvider.BEDROCK
        assert _coerce_provider("vertex") == APIProvider.VERTEX

    def test_coerce_provider_with_invalid_string(self):
        """Test that invalid strings default to ANTHROPIC."""
        result = _coerce_provider("invalid_provider")
        assert result == APIProvider.ANTHROPIC

    def test_coerce_provider_with_empty_string(self):
        """Test that empty string defaults to ANTHROPIC."""
        result = _coerce_provider("")
        assert result == APIProvider.ANTHROPIC

    def test_coerce_provider_case_sensitive(self):
        """Test that provider coercion is case-sensitive."""
        # Uppercase should default to ANTHROPIC
        result = _coerce_provider("ANTHROPIC")
        assert result == APIProvider.ANTHROPIC

    def test_coerce_provider_preserves_enum_type(self):
        """Test that result is always APIProvider type."""
        result = _coerce_provider("kilo")
        assert isinstance(result, APIProvider)

    def test_coerce_provider_with_unknown_value(self):
        """Test with completely unknown value."""
        result = _coerce_provider("some_random_value")
        assert result == APIProvider.ANTHROPIC


class TestValidateAuth:
    """Test the validate_auth function."""

    def test_validate_auth_anthropic_without_key(self):
        """Test validation fails for Anthropic without API key."""
        result = validate_auth(APIProvider.ANTHROPIC, None)
        assert result is not None
        assert "Anthropic API key" in result

    def test_validate_auth_anthropic_with_empty_key(self):
        """Test validation fails for Anthropic with empty key."""
        result = validate_auth(APIProvider.ANTHROPIC, "")
        assert result is not None
        assert "Anthropic API key" in result

    def test_validate_auth_anthropic_with_key(self):
        """Test validation succeeds for Anthropic with key."""
        result = validate_auth(APIProvider.ANTHROPIC, "sk-ant-test-key")
        assert result is None

    def test_validate_auth_kilo_without_key(self):
        """Test validation fails for Kilo without API key."""
        result = validate_auth(APIProvider.KILO, None)
        assert result is not None
        assert "Kilo API key" in result

    def test_validate_auth_kilo_with_key(self):
        """Test validation succeeds for Kilo with key."""
        result = validate_auth(APIProvider.KILO, "kilo-test-key")
        assert result is None

    def test_validate_auth_bedrock_without_credentials(self):
        """Test validation fails for Bedrock without credentials."""
        with patch('boto3.Session') as mock_session_cls:
            mock_session = Mock()
            mock_session.get_credentials.return_value = None
            mock_session_cls.return_value = mock_session

            result = validate_auth(APIProvider.BEDROCK, None)
            assert result is not None
            assert "AWS credentials" in result

    def test_validate_auth_bedrock_with_credentials(self):
        """Test validation succeeds for Bedrock with credentials."""
        with patch('boto3.Session') as mock_session_cls:
            mock_session = Mock()
            mock_credentials = Mock()
            mock_session.get_credentials.return_value = mock_credentials
            mock_session_cls.return_value = mock_session

            result = validate_auth(APIProvider.BEDROCK, None)
            assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_auth_vertex_without_region(self):
        """Test validation fails for Vertex without region."""
        result = validate_auth(APIProvider.VERTEX, None)
        assert result is not None
        assert "CLOUD_ML_REGION" in result

    @patch.dict(os.environ, {"CLOUD_ML_REGION": "us-central1"})
    def test_validate_auth_vertex_without_credentials(self):
        """Test validation fails for Vertex without credentials."""
        with patch('google.auth.default') as mock_default:
            # Create a mock exception class
            class MockDefaultCredentialsError(Exception):
                pass

            # Patch the exception module
            with patch('google.auth.exceptions.DefaultCredentialsError', MockDefaultCredentialsError):
                mock_default.side_effect = MockDefaultCredentialsError()

                result = validate_auth(APIProvider.VERTEX, None)
                assert result is not None
                assert "google cloud credentials" in result

    @patch.dict(os.environ, {"CLOUD_ML_REGION": "us-central1"})
    def test_validate_auth_vertex_with_credentials(self):
        """Test validation succeeds for Vertex with credentials."""
        with patch('google.auth.default') as mock_default:
            mock_credentials = Mock()
            mock_project = "test-project"
            mock_default.return_value = (mock_credentials, mock_project)

            result = validate_auth(APIProvider.VERTEX, None)
            assert result is None

    def test_validate_auth_with_string_provider(self):
        """Test validation works with string provider."""
        result = validate_auth("anthropic", "test-key")
        assert result is None

    def test_validate_auth_invalid_provider_string(self):
        """Test validation with invalid provider string."""
        # Should coerce to ANTHROPIC and check for key
        result = validate_auth("invalid", None)
        assert result is not None


class TestLoadFromStorage:
    """Test the load_from_storage function."""

    def test_load_from_storage_existing_file(self):
        """Test loading from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            test_file = config_dir / "test.txt"
            test_file.write_text("test_data")

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                result = load_from_storage("test.txt")
                assert result == "test_data"

    def test_load_from_storage_nonexistent_file(self):
        """Test loading from nonexistent file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                result = load_from_storage("nonexistent.txt")
                assert result is None

    def test_load_from_storage_empty_file(self):
        """Test loading from empty file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            test_file = config_dir / "empty.txt"
            test_file.write_text("")

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                result = load_from_storage("empty.txt")
                assert result is None

    def test_load_from_storage_whitespace_only(self):
        """Test loading file with only whitespace returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            test_file = config_dir / "whitespace.txt"
            test_file.write_text("   \n\t  ")

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                result = load_from_storage("whitespace.txt")
                assert result is None

    def test_load_from_storage_strips_whitespace(self):
        """Test that loaded data is stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            test_file = config_dir / "data.txt"
            test_file.write_text("  test_data  \n")

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                result = load_from_storage("data.txt")
                assert result == "test_data"

    @patch('computer_use_demo.streamlit.st')
    def test_load_from_storage_handles_exceptions(self, mock_st):
        """Test that exceptions are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Create a file path that will cause an error
            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir / "nonexistent"):
                result = load_from_storage("test.txt")
                assert result is None


class TestSaveToStorage:
    """Test the save_to_storage function."""

    def test_save_to_storage_creates_file(self):
        """Test saving creates file with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                save_to_storage("test.txt", "test_data")

                test_file = config_dir / "test.txt"
                assert test_file.exists()
                assert test_file.read_text() == "test_data"

    def test_save_to_storage_creates_directory(self):
        """Test saving creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "new_dir"

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                save_to_storage("test.txt", "test_data")

                assert config_dir.exists()
                test_file = config_dir / "test.txt"
                assert test_file.exists()

    def test_save_to_storage_overwrites_existing(self):
        """Test saving overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            test_file = config_dir / "test.txt"
            test_file.write_text("old_data")

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                save_to_storage("test.txt", "new_data")

                assert test_file.read_text() == "new_data"

    def test_save_to_storage_sets_permissions(self):
        """Test that saved file has correct permissions (0o600)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                save_to_storage("test.txt", "test_data")

                test_file = config_dir / "test.txt"
                # Check permissions (owner read/write only)
                assert oct(test_file.stat().st_mode)[-3:] == '600'

    def test_save_to_storage_empty_string(self):
        """Test saving empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            with patch('computer_use_demo.streamlit.CONFIG_DIR', config_dir):
                save_to_storage("empty.txt", "")

                test_file = config_dir / "empty.txt"
                assert test_file.exists()
                assert test_file.read_text() == ""

    @patch('computer_use_demo.streamlit.st')
    def test_save_to_storage_handles_exceptions(self, mock_st):
        """Test that exceptions are handled gracefully."""
        # Try to save to a path that will cause an error
        with patch('computer_use_demo.streamlit.CONFIG_DIR', Path("/invalid/path/that/does/not/exist")):
            # Should not raise an exception
            save_to_storage("test.txt", "data")
            # Verify error was written to streamlit
            mock_st.write.assert_called()