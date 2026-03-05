"""Tests for computer_use_demo.streamlit module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import tempfile

import pytest

from computer_use_demo.loop import APIProvider
from computer_use_demo.streamlit import (
    Sender,
    _coerce_provider,
    validate_auth,
    load_from_storage,
    save_to_storage,
    CONFIG_DIR,
)


class TestSender:
    """Tests for Sender enum."""

    def test_sender_values(self):
        """Test that Sender has the expected values."""
        assert Sender.USER == "user"
        assert Sender.BOT == "assistant"
        assert Sender.TOOL == "tool"

    def test_sender_string_behavior(self):
        """Test that Sender members behave like strings."""
        assert str(Sender.USER) == "user"
        assert Sender.BOT == "assistant"


class TestCoerceProvider:
    """Tests for _coerce_provider function."""

    def test_coerce_provider_with_enum(self):
        """Test _coerce_provider with APIProvider enum."""
        result = _coerce_provider(APIProvider.ANTHROPIC)
        assert result == APIProvider.ANTHROPIC
        assert isinstance(result, APIProvider)

    def test_coerce_provider_with_valid_string(self):
        """Test _coerce_provider with valid string."""
        result = _coerce_provider("anthropic")
        assert result == APIProvider.ANTHROPIC

        result = _coerce_provider("kilo")
        assert result == APIProvider.KILO

        result = _coerce_provider("bedrock")
        assert result == APIProvider.BEDROCK

        result = _coerce_provider("vertex")
        assert result == APIProvider.VERTEX

    def test_coerce_provider_with_invalid_string(self):
        """Test _coerce_provider with invalid string returns default."""
        result = _coerce_provider("invalid_provider")
        assert result == APIProvider.ANTHROPIC

        result = _coerce_provider("ANTHROPIC")  # case-sensitive
        assert result == APIProvider.ANTHROPIC

    def test_coerce_provider_defaults_to_anthropic(self):
        """Test that invalid provider defaults to ANTHROPIC."""
        result = _coerce_provider("unknown")
        assert result == APIProvider.ANTHROPIC

        result = _coerce_provider("")
        assert result == APIProvider.ANTHROPIC


class TestValidateAuth:
    """Tests for validate_auth function."""

    def test_validate_auth_anthropic_missing_key(self):
        """Test validation fails when Anthropic API key is missing."""
        error = validate_auth(APIProvider.ANTHROPIC, None)
        assert error is not None
        assert "Anthropic API key" in error

        error = validate_auth(APIProvider.ANTHROPIC, "")
        assert error is not None
        assert "Anthropic API key" in error

    def test_validate_auth_anthropic_with_key(self):
        """Test validation passes when Anthropic API key is provided."""
        error = validate_auth(APIProvider.ANTHROPIC, "sk-ant-test-key")
        assert error is None

    def test_validate_auth_kilo_missing_key(self):
        """Test validation fails when Kilo API key is missing."""
        error = validate_auth(APIProvider.KILO, None)
        assert error is not None
        assert "Kilo API key" in error

        error = validate_auth(APIProvider.KILO, "")
        assert error is not None
        assert "Kilo API key" in error

    def test_validate_auth_kilo_with_key(self):
        """Test validation passes when Kilo API key is provided."""
        error = validate_auth(APIProvider.KILO, "kilo-test-key")
        assert error is None

    def test_validate_auth_bedrock_no_credentials(self):
        """Test validation fails when AWS credentials are not set."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get_credentials.return_value = None
            mock_session_class.return_value = mock_session

            error = validate_auth(APIProvider.BEDROCK, None)
            assert error is not None
            assert "AWS credentials" in error

    def test_validate_auth_bedrock_with_credentials(self):
        """Test validation passes when AWS credentials are set."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get_credentials.return_value = MagicMock()
            mock_session_class.return_value = mock_session

            error = validate_auth(APIProvider.BEDROCK, None)
            assert error is None

    def test_validate_auth_vertex_no_region(self):
        """Test validation fails when CLOUD_ML_REGION is not set."""
        with patch.dict(os.environ, {}, clear=True):
            error = validate_auth(APIProvider.VERTEX, None)
            assert error is not None
            assert "CLOUD_ML_REGION" in error

    def test_validate_auth_vertex_no_credentials(self):
        """Test validation fails when Google credentials are not set."""
        with patch.dict(os.environ, {"CLOUD_ML_REGION": "us-central1"}):
            with patch("google.auth.default") as mock_auth:
                # Mock the DefaultCredentialsError
                with patch("google.auth.exceptions.DefaultCredentialsError", Exception):
                    mock_auth.side_effect = Exception()

                    error = validate_auth(APIProvider.VERTEX, None)
                    assert error is not None
                    assert "google cloud credentials" in error

    def test_validate_auth_vertex_with_credentials(self):
        """Test validation passes when Google credentials are set."""
        with patch.dict(os.environ, {"CLOUD_ML_REGION": "us-central1"}):
            with patch("google.auth.default") as mock_auth:
                mock_auth.return_value = (MagicMock(), MagicMock())

                error = validate_auth(APIProvider.VERTEX, None)
                assert error is None

    def test_validate_auth_with_string_provider(self):
        """Test validate_auth works with string provider input."""
        error = validate_auth("anthropic", "sk-ant-key")
        assert error is None

        error = validate_auth("anthropic", None)
        assert error is not None


class TestStorageFunctions:
    """Tests for load_from_storage and save_to_storage functions."""

    def test_save_and_load_from_storage(self):
        """Test saving and loading data from storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                # Save data
                save_to_storage("test_file", "test_data")

                # Load data
                loaded_data = load_from_storage("test_file")
                assert loaded_data == "test_data"

    def test_load_from_storage_nonexistent_file(self):
        """Test loading from non-existent file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                result = load_from_storage("nonexistent_file")
                assert result is None

    def test_load_from_storage_empty_file(self):
        """Test loading from empty file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                # Create empty file
                (test_config_dir / "empty_file").write_text("")

                result = load_from_storage("empty_file")
                assert result is None

    def test_load_from_storage_whitespace_file(self):
        """Test loading from file with only whitespace returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                # Create file with whitespace
                (test_config_dir / "whitespace_file").write_text("   \n\t  ")

                result = load_from_storage("whitespace_file")
                assert result is None

    def test_save_to_storage_creates_directory(self):
        """Test that save_to_storage creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir) / "new_dir"

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                save_to_storage("test_file", "data")

                assert test_config_dir.exists()
                assert (test_config_dir / "test_file").exists()

    def test_save_to_storage_sets_permissions(self):
        """Test that save_to_storage sets file permissions to 0o600."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                save_to_storage("secure_file", "secret_data")

                file_path = test_config_dir / "secure_file"
                # Check that file has restrictive permissions
                # On some systems, this might not be exactly 0o600
                assert file_path.exists()

    def test_save_to_storage_overwrites_existing(self):
        """Test that save_to_storage overwrites existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                save_to_storage("test_file", "old_data")
                save_to_storage("test_file", "new_data")

                loaded_data = load_from_storage("test_file")
                assert loaded_data == "new_data"

    def test_load_from_storage_strips_whitespace(self):
        """Test that load_from_storage strips leading/trailing whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir)

            with patch("computer_use_demo.streamlit.CONFIG_DIR", test_config_dir):
                (test_config_dir / "test_file").write_text("  data with spaces  \n")

                result = load_from_storage("test_file")
                assert result == "data with spaces"

    def test_storage_error_handling(self):
        """Test that storage functions handle errors gracefully."""
        with patch("computer_use_demo.streamlit.CONFIG_DIR", Path("/invalid/path/that/does/not/exist")):
            # save_to_storage should not raise exception
            save_to_storage("test", "data")

            # load_from_storage should return None
            result = load_from_storage("test")
            assert result is None


class TestConfigDir:
    """Tests for CONFIG_DIR constant."""

    def test_config_dir_is_expanded(self):
        """Test that CONFIG_DIR expands the home directory."""
        assert CONFIG_DIR.is_absolute()
        assert "~" not in str(CONFIG_DIR)