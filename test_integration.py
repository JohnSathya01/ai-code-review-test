#!/usr/bin/env python3
"""
Integration test for the complete AI Code Review Agent system
"""

import sys
import os
import tempfile
import subprocess
from unittest.mock import patch, MagicMock

sys.path.append('.')
from ai_code_review import *

def test_complete_workflow():
    """Test the complete workflow with mock data"""
    print("🧪 Testing Complete AI Code Review Workflow")
    print("=" * 60)
    
    # Mock environment variables
    mock_env = {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'AWS_REGION': 'us-east-1',
        'GITHUB_TOKEN': 'test_token',
        'GITHUB_ACTOR': 'test_actor',
        'GITHUB_REPOSITORY': 'test/repo',
        'GITHUB_SHA': 'abc123',
        'GITHUB_PR_NUMBER': '42'
    }
    
    # Mock Bedrock client
    class MockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
            # Return different responses based on prompt content
            if "review plan" in prompt.lower() or "tasks" in prompt.lower():
                return {
                    "tasks": [
                        {
                            "file": "test.py",
                            "depth": "deep",
                            "checklist_sections": ["Security", "Error Handling"],
                            "reason": "Python file with potential security issues"
                        }
                    ]
                }
            elif "findings" in prompt.lower():
                return {
                    "findings": [
                        {
                            "title": "Missing input validation",
                            "description": "User input should be validated before processing to prevent injection attacks",
                            "severity": "high",
                            "checklist_category": "Security",
                            "file": "test.py",
                            "line": 10,
                            "confidence": 0.9
                        }
                    ]
                }
            elif "final_issues" in prompt.lower():
                return {
                    "final_issues": [
                        {
                            "id": 0,
                            "keep": True,
                            "confidence": 0.9,
                            "severity": "high",
                            "reasoning": "Clear security vulnerability"
                        }
                    ]
                }
            else:
                return {"test": "response"}
    
    # Mock git operations
    def mock_get_git_diff():
        return "A\ttest.py"
    
    def mock_get_file_stats(file_path):
        return (20, 0)  # 20 lines added, 0 removed
    
    def mock_read_file_content(file_path, max_size=None):
        return '''def process_user_input(user_data):
    # This function processes user input without validation
    query = "SELECT * FROM users WHERE id = " + user_data
    return execute_query(query)

def execute_query(query):
    # Execute SQL query
    pass
'''
    
    # Mock GitHub API
    def mock_post_request(*args, **kwargs):
        response = MagicMock()
        response.status_code = 201
        response.text = "Success"
        return response
    
    # Apply all mocks
    with patch.dict(os.environ, mock_env), \
         patch('ai_code_review.get_git_diff', mock_get_git_diff), \
         patch('ai_code_review.get_file_stats', mock_get_file_stats), \
         patch('ai_code_review.read_file_content', mock_read_file_content), \
         patch('ai_code_review.BedrockClient', MockBedrockClient), \
         patch('requests.post', mock_post_request):
        
        try:
            print("Step 1: Testing environment validation...")
            config = validate_startup_environment()
            assert config['AWS_REGION'] == 'us-east-1'
            print("✅ Environment validation passed")
            
            print("\nStep 2: Testing Bedrock client creation...")
            bedrock_client = MockBedrockClient()
            print("✅ Bedrock client created")
            
            print("\nStep 3: Testing changed files extraction...")
            changed_files = get_changed_files()
            assert len(changed_files) == 1
            assert changed_files[0].path == 'test.py'
            assert changed_files[0].change_type == 'added'
            print(f"✅ Found {len(changed_files)} changed files")
            
            print("\nStep 4: Testing Planner Agent...")
            planner = create_planner_agent(bedrock_client)
            planner_output = planner.plan_review(changed_files)
            assert len(planner_output.tasks) == 1
            assert planner_output.tasks[0].file == 'test.py'
            assert planner_output.tasks[0].depth == 'deep'
            print(f"✅ Generated {len(planner_output.tasks)} review tasks")
            
            print("\nStep 5: Testing Reviewer Agent...")
            reviewer = create_reviewer_agent(bedrock_client)
            reviewer_output = reviewer.review_code(planner_output.tasks, changed_files)
            assert len(reviewer_output.findings) == 1
            assert reviewer_output.findings[0].severity == 'high'
            assert reviewer_output.findings[0].checklist_category == 'Security'
            print(f"✅ Found {len(reviewer_output.findings)} potential issues")
            
            print("\nStep 6: Testing Verifier Agent...")
            verifier = create_verifier_agent(bedrock_client)
            verifier_output = verifier.verify_findings(reviewer_output.findings)
            assert len(verifier_output.final_issues) == 1
            assert verifier_output.final_issues[0].severity == 'high'
            print(f"✅ Verified {len(verifier_output.final_issues)} final issues")
            
            print("\nStep 7: Testing GitHub integration...")
            success = post_review_comment(config, verifier_output.final_issues)
            assert success == True
            print("✅ GitHub comments posted successfully")
            
            print("\nStep 8: Testing summary generation...")
            summary = create_review_summary(verifier_output.final_issues)
            print(f"Generated summary: {summary[:100]}...")  # Debug output
            assert "**Total Issues Found:** 1" in summary
            assert "**High:** 1 issues" in summary
            print("✅ Summary generated successfully")
            
            print("\n" + "=" * 60)
            print("🎉 Complete workflow integration test PASSED!")
            print("✅ All components working together correctly")
            
            # Print the summary
            print("\n" + summary)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 Testing Error Handling Scenarios")
    print("=" * 60)
    
    try:
        # Test missing environment variables
        print("Testing missing environment variables...")
        with patch.dict(os.environ, {}, clear=True):
            try:
                validate_startup_environment()
                assert False, "Should have raised EnvironmentError"
            except SystemExit:
                print("✅ Missing environment variables handled correctly")
        
        # Test empty changed files
        print("\nTesting empty changed files...")
        mock_client = MagicMock()
        planner = create_planner_agent(mock_client)
        result = planner.plan_review([])
        assert len(result.tasks) == 0
        print("✅ Empty changed files handled correctly")
        
        # Test empty findings
        print("\nTesting empty findings...")
        verifier = create_verifier_agent(mock_client)
        result = verifier.verify_findings([])
        assert len(result.final_issues) == 0
        print("✅ Empty findings handled correctly")
        
        print("\n✅ All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {str(e)}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 AI Code Review Agent - Integration Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test complete workflow
    if not test_complete_workflow():
        success = False
    
    # Test error handling
    if not test_error_handling():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ AI Code Review Agent is ready for production use")
    else:
        print("❌ Some integration tests failed")
        print("🔧 Please review the errors above and fix issues")
        sys.exit(1)

if __name__ == "__main__":
    main()