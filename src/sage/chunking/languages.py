"""File-extension to language mapping for code-aware chunking."""

from __future__ import annotations

from pathlib import PurePath

__all__ = ["is_code_file", "language_for"]

# Extension -> tree-sitter language name (as used by tree-sitter-language-pack,
# which chonkie's CodeChunker dispatches to).
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".lua": "lua",
    ".zig": "zig",
    ".sh": "bash",
    ".bash": "bash",
    ".css": "css",
    ".html": "html",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def language_for(filename: str) -> str | None:
    """Return the tree-sitter language name for ``filename``, or ``None``."""
    return _EXT_TO_LANG.get(PurePath(filename).suffix.lower())


def is_code_file(filename: str) -> bool:
    """Whether the file should be chunked with the code (AST) chunker."""
    return language_for(filename) is not None
