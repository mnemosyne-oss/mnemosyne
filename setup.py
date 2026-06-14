"""
Mnemosyne - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Single source of truth for version
version_path = Path(__file__).parent / "mnemosyne" / "__init__.py"
_version = "0.1"
for line in version_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("__version__"):
        _version = line.split("=")[1].strip().strip('"').strip("'")
        break

setup(
    name="mnemosyne-memory",
    version=_version,
    author="Abdias J",
    author_email="1641797+AxDSan@users.noreply.github.com",
    description="The Universal Memory Layer for Any AI Agent — Zero-Dependency, Sub-Millisecond, Fully Private",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AxDSan/mnemosyne",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    keywords=[
        "ai",
        "memory",
        "sqlite",
        "agent",
        "llm",
        "context",
        "embeddings",
        "vector-store",
        "honcho",
        "zep",
        "mcp",
        "openwebui",
        "openclaw",
        "cursor",
        "claude-code",
        "codex",
        "hermes",
        "model-context-protocol",
    ],
    project_urls={
        "Bug Reports": "https://github.com/AxDSan/mnemosyne/issues",
        "Source": "https://github.com/AxDSan/mnemosyne",
        "Documentation": "https://github.com/AxDSan/mnemosyne/blob/main/docs/README.md",
    },
    extras_require={
        "llm": ["ctransformers>=0.2.27", "llama-cpp-python>=0.2.0", "huggingface-hub>=0.20"],
        "embeddings": ["fastembed>=0.3.0", "sqlite-vec>=0.1.0"],
        "mcp": ["mcp>=1.0.0; python_version >= '3.10'", "anyio>=4.0; python_version >= '3.10'"],
        "openclaw": ["openclaw>=0.1.0; python_version >= '3.10'"],
        "test": ["pytest>=7.0"],
        "all": ["ctransformers>=0.2.27", "llama-cpp-python>=0.2.0", "huggingface-hub>=0.20", "fastembed>=0.3.0", "sqlite-vec>=0.1.0", "mcp>=1.0.0; python_version >= '3.10'", "anyio>=4.0; python_version >= '3.10'"],
        "dev": ["pytest>=7.0", "build", "twine"],
    },
    entry_points={
        "console_scripts": [
            "mnemosyne-install=mnemosyne.install:install",
            "mnemosyne-uninstall=mnemosyne.install:uninstall",
            "mnemosyne=mnemosyne.cli:run_cli",
            "mnemosyne-browser=mnemosyne.integrations.memory_browser:main",
            "mnemosyne-auto-save=mnemosyne.integrations.auto_save_openwebui:main",
        ],
    },
)
