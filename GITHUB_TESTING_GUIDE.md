# 🚀 GitHub Testing Guide - AI Code Review Agent

## Quick Start (5 Minutes)

### Step 1: Create GitHub Repository
1. Go to [GitHub.com](https://github.com) and click "New repository"
2. Name: `ai-code-review-test` 
3. Make it **Public** (required for free GitHub Actions)
4. **Don't** initialize with README
5. Click "Create repository"

### Step 2: Push Your Code
```bash
# Add your GitHub repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ai-code-review-test.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Test Immediately (No AWS Required!)
The system will automatically work in **mock mode** if you don't have AWS credentials set up.

## 🎭 Mock Mode Testing (No AWS Required)

The system automatically detects if AWS credentials are missing and switches to mock mode with realistic AI responses.

### Test 1: Push Test Files
```bash
# The test files are already committed, just push them
git push
```

Go to your repository → **Actions** tab to see the AI Code Review running!

### Test 2: Create a Pull Request
```bash
# Create a new branch with more issues
git checkout -b feature/add-crypto-issues

# Create a file with crypto vulnerabilities
cat > crypto_issues.py << 'EOF'
import hashlib
import random

def hash_password(password):
    # Weak hash function - security issue
    return hashlib.md5(password.encode()).hexdigest()

def generate_token():
    # Weak random number generation - security issue  
    return str(random.randint(1000, 9999))

def verify_user(username, password_hash):
    # Hardcoded admin credentials - security issue
    if username == "admin" and password_hash == "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8":
        return True
    return False
EOF

git add crypto_issues.py
git commit -m "Add crypto functions with security vulnerabilities"
git push -u origin feature/add-crypto-issues
```

Now go to GitHub and create a Pull Request from this branch. The AI will analyze it and post detailed comments!

## 🔧 Real AWS Testing (Optional)

If you want to test with real AWS Bedrock Claude:

### Step 1: Set Up AWS Bedrock
1. Go to [AWS Console](https://console.aws.amazon.com) → Bedrock
2. Navigate to "Model access"
3. Request access to "Claude 3.5 Sonnet" (usually instant approval)
4. Create IAM user with this policy:

```json
{
  "Version": "2012-10-17", 
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
  ]
}
```

### Step 2: Add GitHub Secrets
1. Go to your repository → Settings → Secrets and variables → Actions
2. Add these secrets:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `AWS_REGION`: `us-east-1` (or your region)

### Step 3: Test with Real AI
Push any code change and the system will automatically use real Claude AI instead of mock responses.

## 📊 Expected Results

### Mock Mode Results
The AI should detect and comment on:

**test_security.py:**
- 🚨 **SQL Injection Vulnerability** (HIGH)
- 🚨 **Hardcoded API Key** (HIGH) 
- ⚠️ **Missing Input Validation** (MEDIUM)

**file_handler.py:**
- 🚨 **Path Traversal Vulnerability** (HIGH)
- 🚨 **Command Injection** (HIGH)
- ⚠️ **Missing Exception Handling** (MEDIUM)

**crypto_issues.py:**
- 🚨 **Weak Hash Function (MD5)** (HIGH)
- ⚠️ **Weak Random Generation** (MEDIUM)
- 🚨 **Hardcoded Credentials** (HIGH)

### GitHub Actions Output
```
🤖 Running AI Code Review with mock responses for demo...
✅ Found 3 changed files to review
✅ Generated 3 review tasks  
✅ Found 8 potential issues
✅ Verified 6 final issues
🎉 Posted 6 review comments to GitHub
```

## 🔍 Monitoring Your Tests

### 1. GitHub Actions Tab
- Go to your repository → **Actions**
- Click on the latest workflow run
- Expand "AI Code Review Agent" to see detailed logs

### 2. Pull Request Comments  
- The AI posts detailed comments on each issue found
- Comments include severity, category, confidence, and recommendations

### 3. Issues Tab (Future Feature)
- High-severity findings can automatically create GitHub Issues
- Helps track security vulnerabilities across your codebase

## 🎯 Testing Scenarios

### Scenario 1: Security Vulnerabilities
```python
# Create sql_injection.py
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection!
    return db.execute(query)
```

### Scenario 2: Error Handling Issues
```python  
# Create error_handling.py
def process_file(filename):
    file = open(filename, 'r')  # No exception handling!
    data = file.read()
    return data  # File never closed!
```

### Scenario 3: Performance Issues
```python
# Create performance.py  
def find_user(users, target_id):
    for user in users:  # O(n) search instead of hash lookup
        if user['id'] == target_id:
            return user
    return None
```

## 🐛 Troubleshooting

### "No changes detected"
- Make sure you're pushing actual file changes
- Check that files aren't too large (>1MB limit)
- Verify git diff is working: `git diff HEAD~1`

### "GitHub Actions not running"
- Ensure repository has Actions enabled (Settings → Actions)
- Check if workflow file is in `.github/workflows/`
- Verify the workflow syntax is correct

### "No PR comments appearing"
- Check Actions logs for errors
- Verify `GITHUB_TOKEN` has proper permissions
- Ensure the PR has actual code changes

### "Mock mode not working"
- Check that `ai_code_review_mock.py` exists
- Verify Python dependencies are installed
- Look at Actions logs for specific errors

## 🎉 Success Indicators

You'll know it's working when you see:

1. ✅ **GitHub Actions runs successfully** (green checkmark)
2. 🤖 **AI comments appear on PRs** with detailed analysis
3. 📊 **Summary statistics** in the Actions logs
4. 🚨 **Security issues flagged** with appropriate severity

## Next Steps

Once testing is complete:
1. **Deploy to production repositories** by copying the files
2. **Set up real AWS Bedrock** for production-grade AI analysis  
3. **Customize the checklist** for your specific needs
4. **Configure issue creation** for high-severity findings

---

**🚀 Ready to test? Push your code and watch the AI review it in real-time!**