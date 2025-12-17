# GitHub Setup Guide

This guide will walk you through publishing your Agentic AI Trader project to GitHub.

## Pre-Commit Checklist

✅ **Files created:**
- `.gitignore` - Excludes sensitive/copyrighted files
- `.env.example` - Template for environment variables
- `.gitkeep` files - Preserve empty directory structure

✅ **Files excluded (won't be pushed to GitHub):**
- `.env` - Your actual API keys (PROTECTED)
- `data/books/*.pdf` - Trading books (COPYRIGHT PROTECTED)
- `*.db` files - Cached data (too large)
- `ml_models/saved_models/*.pkl` - Trained models (too large)
- `test_*.py` - Test files (not needed by users)
- `__pycache__/` - Python cache
- `.venv*/` - Virtual environments

## Step-by-Step GitHub Setup

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Choose a name: `agentic-ai-trader` (or whatever you prefer)
3. Description: `RAG-enhanced AI trading agent with ML predictions, regime detection, and sentiment analysis`
4. Make it **Public** or **Private** (your choice)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 2: Initialize Local Git Repository

```powershell
# Navigate to your project
cd C:\Users\Andrew\projects\agentic_ai_trader

# Initialize git (if not already done)
git init

# Add all files (gitignore will exclude sensitive ones)
git add .

# Check what will be committed
git status
```

**Review the output!** Make sure you don't see:
- `.env` file
- PDF files in `data/books/`
- Any `*.db` files
- Your API keys

### Step 3: Make Initial Commit

```powershell
# Commit your code
git commit -m "Initial commit: RAG-enhanced trading agent with ML predictions"
```

### Step 4: Connect to GitHub

```powershell
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/agentic-ai-trader.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 5: Verify on GitHub

1. Go to your repository: `https://github.com/YOUR_USERNAME/agentic-ai-trader`
2. Check that:
   - ✅ Code is there
   - ✅ No `.env` file visible
   - ✅ No PDF files in `data/books/`
   - ✅ `.gitignore` is working
   - ✅ README.md displays nicely

## Troubleshooting

### Problem: "git: command not found"

**Solution:** Install Git for Windows
1. Download: https://git-scm.com/download/win
2. Install with default options
3. Restart PowerShell
4. Verify: `git --version`

### Problem: ".env file is showing up in git status"

**Solution:**
```powershell
# Make sure .env is in .gitignore
cat .gitignore | Select-String ".env"

# If it's there but still showing, remove from git tracking:
git rm --cached .env
git commit -m "Remove .env from tracking"
```

### Problem: "PDF files are too large to push"

**Solution:** They should already be excluded by `.gitignore`
```powershell
# Verify PDFs are ignored
git status | Select-String "pdf"

# If any show up, they shouldn't. Check .gitignore:
cat .gitignore | Select-String "pdf"
```

### Problem: "Authentication failed"

**Solution:** Use Personal Access Token (PAT)
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (all)
4. Copy the token
5. When pushing, use token as password

Or set up SSH keys (more secure, one-time setup):
```powershell
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~\.ssh\id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Then use SSH remote instead:
git remote set-url origin git@github.com:YOUR_USERNAME/agentic-ai-trader.git
```

## Future Updates

After making changes to your project:

```powershell
# Check what changed
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add sentiment analysis for 50 S&P 500 stocks"

# Push to GitHub
git push
```

## Best Practices

### Commit Messages

Good:
- `Add ML prediction tool with missing features handling`
- `Fix consensus agreement calculation in ml_prediction_tool.py`
- `Enhance ML formatter with emoji and visual hierarchy`

Bad:
- `Update`
- `Fix bug`
- `Changes`

### What to Commit

✅ **DO commit:**
- Source code (`.py` files)
- Documentation (`.md` files)
- Configuration templates (`.env.example`)
- Requirements (`requirements.txt`)

❌ **DON'T commit:**
- API keys (`.env`)
- Large data files (`.db`, `.csv`, `.pkl`)
- Copyrighted content (PDFs)
- Test outputs
- Virtual environments

### Branching (Optional but Recommended)

For major changes, use branches:

```powershell
# Create new branch for feature
git checkout -b feature/sentiment-batch-3

# Make changes, commit
git add .
git commit -m "Add sentiment data for batch 3 stocks"

# Push branch
git push -u origin feature/sentiment-batch-3

# On GitHub: Create Pull Request → Merge when ready
```

## Security Checklist

Before pushing, verify:

- [ ] `.env` file is NOT in git: `git ls-files | Select-String ".env"`
- [ ] No API keys in code: `git grep -i "sk-" "*.py"`
- [ ] No PDF files: `git ls-files | Select-String ".pdf"`
- [ ] No database files: `git ls-files | Select-String ".db"`
- [ ] `.gitignore` is working: `git status` shows clean

## Public vs Private Repository

**Public:**
- ✅ Great for portfolio/resume
- ✅ Others can learn from your code
- ✅ Community contributions possible
- ❌ Anyone can see your code (not your API keys though!)

**Private:**
- ✅ Code stays confidential
- ✅ Still accessible to you from anywhere
- ❌ Not visible on your GitHub profile
- ❌ Can't share easily with potential employers

**Recommendation:** Start **private**, make **public** later when ready

## Helpful Commands

```powershell
# Check git status
git status

# View commit history
git log --oneline

# See what files are tracked
git ls-files

# See what's ignored
git status --ignored

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes
git reset --hard HEAD

# Create .gitignore from template
# (Already done for you!)
```

## Getting Help

- Git documentation: https://git-scm.com/doc
- GitHub guides: https://guides.github.com/
- Pro Git book (free): https://git-scm.com/book/en/v2

## Next Steps After Pushing

1. **Add a LICENSE** (MIT, Apache 2.0, GPL, etc.)
2. **Add badges** to README (build status, license, etc.)
3. **Add CONTRIBUTING.md** if you want contributions
4. **Set up GitHub Actions** for CI/CD (optional)
5. **Add topics/tags** on GitHub for discoverability

---

**Ready to push?** Start with Step 1 above! 🚀
