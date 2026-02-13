"""
Install git hooks by copying from hooks/ into .git/hooks/
Run once: python setup_hooks.py
"""

import shutil
from pathlib import Path

HOOKS_DIR = Path(__file__).parent / "hooks"
GIT_HOOKS_DIR = Path(__file__).parent / ".git" / "hooks"

if not GIT_HOOKS_DIR.exists():
    print("[ERROR] .git/hooks/ not found. Is this a git repository?")
    exit(1)

installed = 0
for hook_file in HOOKS_DIR.iterdir():
    if hook_file.name.startswith("."):
        continue
    dest = GIT_HOOKS_DIR / hook_file.name
    shutil.copy2(hook_file, dest)
    print(f"  Installed: {hook_file.name} -> {dest}")
    installed += 1

print(f"\n[OK] {installed} hook(s) installed.")
