# GitHub Testing Setup Guide

## Step 1: Create GitHub Repository

1. Go to GitHub.com and create a new repository
2. Name it something like `ai-code-review-test`
3. Make it **public** (required for GitHub Actions on free accounts)
4. Don't initialize with README (we already have files)

## Step 2: Push Code to GitHub

Run these commands in your terminal:

```bash
# Add your GitHub repository as remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ai-code-review-test.git

# Push the code
git branch -M main
git push -u origin main
```

## Step 3: Set Up AWS Bedrock (Required for Testing)

### Option A: Use Real AWS Account (Recommended)
1. Go to AWS Console → Bedrock
2. Navigate to "Model access" 
3. Request access to "Claude 3.5 Sonnet" model
4. Create IAM user with this policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
  ]
}
```

### Option B: Use Mock Testing (For Demo Only)
If you don't have AWS access, I can modify the workflow to use mock responses for testing.

## Step 4: Configure GitHub Secrets

1. Go to your repository → Settings → Secrets and variables → Actions
2. Add these repository secrets:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key  
   - `AWS_REGION`: `us-east-1` (or your preferred region)

## Step 5: Test the System

### Test 1: Push with Issues
1. Create a new file with security issues:
```python
# test_security.py
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    return execute_sql(query)  # SQL injection vulnerability!
```

2. Commit and push:
```bash
git add test_security.py
git commit -m "Add login function with security issues"
git push
```

### Test 2: Create Pull Request
1. Create a new branch:
```bash
git checkout -b feature/add-file-handler
```

2. Add problematic code:
```python
# file_handler.py
def read_user_file(filename):
    path = "/uploads/" + filename  # Path traversal vulnerability!
    with open(path, 'r') as f:    # No error handling!
        return f.read()
```

3. Push and create PR:
```bash
git add file_handler.py
git commit -m "Add file handler with vulnerabilities"
git push -u origin feature/add-file-handler
```

4. Go to GitHub and create a Pull Request from this branch

## Step 6: Monitor Results

1. **Check GitHub Actions**: Go to Actions tab to see the workflow running
2. **Check PR Comments**: The AI should post detailed security findings
3. **Check Issues**: If configured, issues will be created for high-severity findings

## Expected Results

The AI Code Review Agent should:
- ✅ Detect SQL injection vulnerability
- ✅ Detect path traversal vulnerability  
- ✅ Detect missing error handling
- ✅ Post detailed comments with severity levels
- ✅ Provide actionable recommendations

## Troubleshooting

### If GitHub Actions Fails:
1. Check the Actions tab for error logs
2. Verify AWS credentials are correct
3. Ensure Bedrock model access is approved
4. Check if repository has Actions enabled

### If No Comments Appear:
1. Verify the workflow has `pull-requests: write` permission
2. Check if the PR has actual code changes
3. Look at the Actions logs for any errors

### If AWS Bedrock Fails:
1. Verify model access is approved (can take a few minutes)
2. Check IAM permissions
3. Ensure you're using the correct region
4. Try a different AWS region if needed

## Mock Testing Option

If you want to test without AWS, I can create a mock version that simulates the AI responses for demonstration purposes.