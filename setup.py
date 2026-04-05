"""
Mnemosyne - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="mnemosyne-memory",
    version="1.0.0",
    author="FluxSpeak AI",
    author_email="aj@fluxspeak.ai",
    description="The Zero-Dependency, Sub-Millisecond AI Memory System",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
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
    ],
    project_urls={
        "Bug Reports": "https://github.com/AxDSan/mnemosyne/issues",
        "Source": "https://github.com/AxDSan/mnemosyne",
        "Documentation": "https://github.com/AxDSan/mnemosyne/blob/main/README.md",
    },
)
