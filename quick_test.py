#!/usr/bin/env python3
"""
Quick test script to verify the AI Code Review Agent works locally
"""

import os
import sys
from unittest.mock import patch

# Test the mock version locally
def test_mock_version():
    print("🧪 Testing AI Code Review Agent (Mock Version)")
    print("=" * 50)
    
    # Set up mock environment
    mock_env = {
        'USE_MOCK_AI': 'true',
        'AWS_ACCESS_KEY_ID': 'mock_key',
        'AWS_SECRET_ACCESS_KEY': 'mock_secret', 
        'AWS_REGION': 'us-east-1',
        'GITHUB_TOKEN': 'mock_token',
        'GITHUB_ACTOR': 'test_user',
        'GITHUB_REPOSITORY': 'test/ai-code-review',
        'GITHUB_SHA': 'abc123',
        'GITHUB_PR_NUMBER': '1'
    }
    
    # Mock git operations to use our test files
    def mock_get_git_diff():
        return "A\ttest_security.py\nA\tfile_handler.py"
    
    def mock_get_file_stats(file_path):
        if file_path == "test_security.py":
            return (25, 0)
        elif file_path == "file_handler.py":
            return (30, 0)
        return (0, 0)
    
    def mock_read_file_content(file_path, max_size=None):
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except:
            return ""
    
    def mock_post_request(*args, **kwargs):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.status_code = 201
        response.text = "Success"
        return response
    
    # Apply mocks and run
    with patch.dict(os.environ, mock_env), \
         patch('ai_code_review.get_git_diff', mock_get_git_diff), \
         patch('ai_code_review.get_file_stats', mock_get_file_stats), \
         patch('ai_code_review.read_file_content', mock_read_file_content), \
         patch('requests.post', mock_post_request):
        
        try:
            # Import and run the mock version
            import ai_code_review_mock
            print("✅ Mock AI Code Review completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_mock_version()
    if success:
        print("\n🎉 Local test passed! Ready for GitHub testing.")
    else:
        print("\n❌ Local test failed. Check the errors above.")
        sys.exit(1)