"""
scan_repo.py
Prints a tree of the repo work folder, flags anything that should be gitignored.
Run from anywhere: python scan_repo.py "D:\Projects\STORM"
Or drop it in the folder and run: python scan_repo.py

Flag lists below are intentionally generic. If you have project-specific files
or terms you want auto-flagged before a public push, create a local
`.scan_repo_private.txt` next to this script (one pattern per line, not
tracked by git) and it will be merged in automatically. Keeping private
identifiers out of this script's own source is the point -- a public repo
scanner script should not itself be the thing that leaks what it's protecting.
"""

import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

# Patterns that should generally be blocked by .gitignore
SHOULD_IGNORE = {
    # dirs
    "__pycache__", ".vscode", ".idea", "baks", "dist", "build",
    "node_modules", ".git", "output", "outputs",
    # file extensions
    ".pyc", ".pyd", ".pyo", ".egg-info",
    ".rar", ".zip", ".7z",
    ".DS_Store", "Thumbs.db",
}

# Generic markers that commonly indicate an internal-only doc.
# Project-specific names/filenames go in .scan_repo_private.txt instead --
# see module docstring.
INTERNAL_FLAGS = {
    "_Internal_Reference", "_REPO_INVENTORY",
    "INTERNAL", "_private", "_SECRET",
}


def _load_local_overrides():
    """
    Merge in project-specific patterns from .scan_repo_private.txt if present.
    This file should be in .gitignore -- it's the place for real internal
    filenames/project names without baking them into tracked source.
    """
    override_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_repo_private.txt")
    if not os.path.exists(override_path):
        return
    try:
        with open(override_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                SHOULD_IGNORE.add(line)
    except Exception:
        pass


_load_local_overrides()

# Extensions we care about for the listing
SHOW_EXTS = {
    ".py", ".lua", ".hpp", ".h", ".cpp", ".md", ".txt", ".toml",
    ".yaml", ".yml", ".json", ".bat", ".sh", ".svg", ".png",
    ".safetensors", ".gguf", ".gitignore", ".cfg", ".ini",
}

def flag(name):
    n = name.lower()
    for pat in SHOULD_IGNORE:
        if n == pat.lower() or n.endswith(pat.lower()):
            return "  ← GITIGNORE"
    for pat in INTERNAL_FLAGS:
        if pat.lower() in n:
            return "  ← INTERNAL -- DO NOT PUSH"
    return ""

def scan(path, prefix="", depth=0):
    if depth > 6:
        return
    try:
        entries = sorted(os.scandir(path), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return

    dirs  = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    for i, entry in enumerate(dirs + files):
        is_last = (i == len(dirs) + len(files) - 1)
        connector = "└── " if is_last else "├── "
        ext = os.path.splitext(entry.name)[1].lower()
        f = flag(entry.name)

        if entry.is_dir():
            print(f"{prefix}{connector}{entry.name}/{f}")
            if not f:  # don't recurse into ignored dirs
                extension = "    " if is_last else "│   "
                scan(entry.path, prefix + extension, depth + 1)
        else:
            # show all files but dim ones we don't recognise
            size = os.path.getsize(entry.path)
            size_str = f"{size/1024:.0f}KB" if size > 1024 else f"{size}B"
            if ext in SHOW_EXTS or f:
                print(f"{prefix}{connector}{entry.name}  [{size_str}]{f}")
            else:
                print(f"{prefix}{connector}{entry.name}  [{size_str}]  (unknown ext)")

print(f"\n{'='*60}")
print(f"REPO SCAN: {ROOT}")
print(f"{'='*60}\n")
scan(ROOT)
print(f"\n{'='*60}")
print("Legend:")
print("  ← GITIGNORE          : should be in .gitignore, not committed")
print("  ← INTERNAL           : private doc, must not be pushed")
print("  (unknown ext)         : check manually")
print(f"{'='*60}\n")
