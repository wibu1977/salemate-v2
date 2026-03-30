"""
Tests for configuration validation.
"""
import os
from pathlib import Path


def test_directory_structure_exists():
    """Test that all required directories exist."""
    base_path = Path(__file__).parent.parent
    required_dirs = [
        "app",
        "app/db",
        "app/models",
        "app/routers",
        "app/services",
        "app/utils",
        "tests",
        "supabase",
        "supabase/migrations",
    ]

    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Directory {dir_path} does not exist"
        assert full_path.is_dir(), f"{dir_path} is not a directory"


def test_init_files_exist():
    """Test that all __init__.py files exist."""
    base_path = Path(__file__).parent.parent
    required_init_files = [
        "app/__init__.py",
        "app/db/__init__.py",
        "app/models/__init__.py",
        "app/routers/__init__.py",
        "app/services/__init__.py",
        "app/utils/__init__.py",
        "tests/__init__.py",
    ]

    for init_file in required_init_files:
        full_path = base_path / init_file
        assert full_path.exists(), f"Init file {init_file} does not exist"
        assert full_path.is_file(), f"{init_file} is not a file"


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists."""
    base_path = Path(__file__).parent.parent
    pyproject = base_path / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml does not exist"

    # Verify content
    content = pyproject.read_text()
    assert "[project]" in content, "pyproject.toml missing [project] section"
    assert "fastapi" in content, "pyproject.toml missing fastapi dependency"
    assert "pytest" in content, "pyproject.toml missing pytest in dev dependencies"


def test_env_example_exists():
    """Test that .env.example exists."""
    base_path = Path(__file__).parent.parent
    env_example = base_path / ".env.example"
    assert env_example.exists(), ".env.example does not exist"

    # Verify content
    content = env_example.read_text()
    assert "SUPABASE_URL" in content, ".env.example missing SUPABASE_URL"
    assert "GEMINI_API_KEY" in content, ".env.example missing GEMINI_API_KEY"
