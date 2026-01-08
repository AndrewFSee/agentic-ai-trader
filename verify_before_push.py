"""
Pre-commit verification script
Run this before pushing to GitHub to ensure no sensitive data will be exposed.
"""

import os
import subprocess
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists."""
    return Path(filepath).exists()

def run_git_command(command):
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 80)
    print("PRE-COMMIT VERIFICATION")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # Check 1: .gitignore exists
    print("\n[1/8] Checking .gitignore exists...")
    if check_file_exists(".gitignore"):
        print("  [OK] .gitignore found")
    else:
        print("  [ERROR] .gitignore not found!")
        issues.append(".gitignore is missing")
    
    # Check 2: .env.example exists
    print("\n[2/8] Checking .env.example exists...")
    if check_file_exists(".env.example"):
        print("  [OK] .env.example found")
    else:
        print("  [WARN] .env.example not found")
        warnings.append(".env.example is missing (recommended)")
    
    # Check 3: .env is NOT tracked
    print("\n[3/8] Checking .env is not tracked...")
    success, stdout, stderr = run_git_command("git ls-files .env")
    if stdout.strip() == "":
        print("  [OK] .env is not tracked")
    else:
        print("  [ERROR] .env IS TRACKED - YOUR API KEYS WILL BE EXPOSED!")
        issues.append(".env file is tracked (contains API keys)")
    
    # Check 4: No PDF files tracked
    print("\n[4/8] Checking no PDF files tracked...")
    success, stdout, stderr = run_git_command("git ls-files | findstr /i .pdf")
    if stdout.strip() == "":
        print("  [OK] No PDF files tracked")
    else:
        print(f"  [ERROR] PDF files are tracked (copyright risk):")
        for line in stdout.strip().split('\n'):
            print(f"    - {line}")
        issues.append("PDF files are tracked")
    
    # Check 5: No .db files tracked
    print("\n[5/8] Checking no database files tracked...")
    success, stdout, stderr = run_git_command("git ls-files | findstr /i .db")
    if stdout.strip() == "":
        print("  [OK] No .db files tracked")
    else:
        print(f"  [WARN] Database files are tracked (large files):")
        for line in stdout.strip().split('\n'):
            print(f"    - {line}")
        warnings.append("Database files are tracked (may be too large)")
    
    # Check 6: No API keys in tracked files
    print("\n[6/8] Checking no API keys in code...")
    # Check for actual API key patterns, not just "sk-" which has false positives
    api_key_patterns = [
        ('git grep -E "sk-[a-zA-Z0-9]{20,}" -- "*.py"', "OpenAI API key"),
        ('git grep -E "AKIA[A-Z0-9]{16}" -- "*.py"', "AWS key"),
        ('git grep -E "AIza[0-9A-Za-z\\-_]{35}" -- "*.py"', "Google API key"),
    ]
    api_key_found = False
    for pattern, key_type in api_key_patterns:
        success, stdout, stderr = run_git_command(pattern)
        if stdout.strip() and "os.getenv" not in stdout and ".env" not in stdout:
            print(f"  [ERROR] Possible {key_type} found:")
            for line in stdout.strip().split('\n')[:3]:
                print(f"    - {line}")
            api_key_found = True
            issues.append(f"Possible {key_type} in code")
    
    if not api_key_found:
        print("  [OK] No API keys found in Python files")
    
    # Check 7: No large pickle files
    print("\n[7/8] Checking no large model files tracked...")
    success, stdout, stderr = run_git_command("git ls-files | findstr /i .pkl")
    if stdout.strip() == "":
        print("  [OK] No .pkl files tracked")
    else:
        print(f"  [WARN] Model files tracked (may be too large):")
        for line in stdout.strip().split('\n')[:5]:
            print(f"    - {line}")
        warnings.append("Model .pkl files are tracked (may be too large)")
    
    # Check 8: Git is initialized
    print("\n[8/8] Checking git is initialized...")
    if check_file_exists(".git"):
        print("  [OK] Git repository initialized")
    else:
        print("  [WARN] Git not initialized yet")
        warnings.append("Run 'git init' to initialize repository")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if not issues and not warnings:
        print("\n[SUCCESS] All checks passed! Safe to push to GitHub.")
        print("\nNext steps:")
        print("  1. git add .")
        print("  2. git commit -m 'Initial commit'")
        print("  3. git remote add origin <your-repo-url>")
        print("  4. git push -u origin main")
        return 0
    
    if issues:
        print(f"\n[ERROR] {len(issues)} critical issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n[ACTION REQUIRED] Fix these issues before pushing!")
        
        if ".env file is tracked" in issues:
            print("\nTo remove .env from git:")
            print("  git rm --cached .env")
            print("  git commit -m 'Remove .env from tracking'")
    
    if warnings:
        print(f"\n[WARN] {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print("\nThese are not critical but should be reviewed.")
    
    return 1 if issues else 0

if __name__ == "__main__":
    exit(main())
