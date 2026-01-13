# 🔧 Manual GitHub Actions Workflow Setup

Since your GitHub token doesn't have the `workflow` scope, we need to add the GitHub Actions workflow manually through the web interface.

## Step 1: Create the Workflow File

1. Go to your repository: https://github.com/JohnSathya01/ai-code-review-test
2. Click on **"Actions"** tab
3. Click **"set up a workflow yourself"** or **"New workflow"**
4. Replace the default content with this:

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [main, master, develop]
  push:
    branches: [main, master, develop]

jobs:
  ai-code-review:
    name: AI Code Review Agent
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      pull-requests: write
      issues: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch full history for git diff
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install boto3 requests
      
      - name: Set up Git configuration
        run: |
          git config --global user.name "AI Code Review Agent"
          git config --global user.email "ai-code-review@github.com"
      
      - name: Extract PR number for pull request events
        if: github.event_name == 'pull_request'
        run: |
          echo "GITHUB_PR_NUMBER=${{ github.event.number }}" >> $GITHUB_ENV
      
      - name: Check if AWS credentials are available
        id: check_aws
        run: |
          if [ -n "${{ secrets.AWS_ACCESS_KEY_ID }}" ] && [ -n "${{ secrets.AWS_SECRET_ACCESS_KEY }}" ]; then
            echo "aws_available=true" >> $GITHUB_OUTPUT
            echo "✅ AWS credentials found - using real AI analysis"
          else
            echo "aws_available=false" >> $GITHUB_OUTPUT
            echo "⚠️ AWS credentials not found - using mock AI for demo"
          fi
      
      - name: Run AI Code Review (Real AWS)
        if: steps.check_aws.outputs.aws_available == 'true'
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_REGION }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_ACTOR: ${{ github.actor }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          GITHUB_SHA: ${{ github.sha }}
          GITHUB_REF: ${{ github.ref }}
        run: |
          echo "🤖 Running AI Code Review with real AWS Bedrock Claude..."
          python ai_code_review.py
      
      - name: Run AI Code Review (Mock Mode)
        if: steps.check_aws.outputs.aws_available == 'false'
        env:
          USE_MOCK_AI: 'true'
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_ACTOR: ${{ github.actor }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          GITHUB_SHA: ${{ github.sha }}
          GITHUB_REF: ${{ github.ref }}
        run: |
          echo "🎭 Running AI Code Review with mock responses for demo..."
          python ai_code_review_mock.py
      
      - name: Upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: ai-code-review-logs
          path: |
            *.log
            /tmp/ai-code-review-*.log
          retention-days: 7
```

5. Name the file: `ai-code-review.yml`
6. Click **"Commit changes..."**
7. Add commit message: "Add AI Code Review workflow"
8. Click **"Commit changes"**

## Step 2: Test It Immediately!

Once you've added the workflow:

1. **It will run automatically** on the commit you just made
2. Go to **Actions** tab to see it running
3. Click on the workflow run to see detailed logs

## Step 3: Test with Pull Request

Create a test PR to see the AI in action:

```bash
# In your local repository
git pull origin main  # Get the workflow file

git checkout -b feature/test-vulnerabilities

# Create a file with security issues
cat > security_test.py << 'EOF'
import os
import sqlite3

def login(username, password):
    # SQL Injection vulnerability!
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    conn = sqlite3.connect('users.db')
    return conn.execute(query).fetchone()

def read_file(filename):
    # Path traversal vulnerability!
    path = "/uploads/" + filename
    with open(path, 'r') as f:  # No error handling!
        return f.read()

# Hardcoded secret!
API_KEY = "sk-1234567890abcdef"
EOF

git add security_test.py
git commit -m "Add security test file with vulnerabilities"
git push -u origin feature/test-vulnerabilities
```

Then go to GitHub and create a Pull Request. **The AI will automatically analyze it and post detailed security comments!**

## 🎯 Expected Results

You should see:
- ✅ **GitHub Actions runs successfully**
- 🤖 **AI comments on the PR** pointing out:
  - 🚨 SQL Injection vulnerability
  - 🚨 Path traversal vulnerability  
  - 🚨 Hardcoded API key
  - ⚠️ Missing error handling
- 📊 **Detailed analysis logs** in the Actions tab

## 🎉 You're Ready!

Once the workflow is added, the AI Code Review Agent will automatically:
- Run on every push to main
- Run on every pull request
- Post intelligent security analysis
- Work in mock mode (no AWS required for testing)

**Go add that workflow and watch the magic happen!** 🚀