#!/usr/bin/env python3
"""
Test script for Reviewer Agent implementation
"""

import sys
import os
sys.path.append('.')

from ai_code_review import *

def test_reviewer_data_models():
    """Test that Reviewer Agent data models work correctly"""
    print("Testing Reviewer Agent data models...")
    
    # Test Finding creation
    test_finding = Finding(
        title="Potential SQL injection vulnerability",
        description="User input is directly concatenated into SQL query without sanitization",
        severity="high",
        checklist_category="Security",
        file="test.py",
        line=42,
        confidence=0.9
    )
    print("✅ Finding creation successful")
    
    # Test ReviewerOutput creation
    test_output = ReviewerOutput(findings=[test_finding])
    print("✅ ReviewerOutput creation successful")
    
    return test_finding, test_output

def test_reviewer_logic():
    """Test Reviewer Agent logic without Claude"""
    print("\nTesting Reviewer Agent logic...")
    
    # Create mock bedrock client
    class MockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
            # Return a mock response that matches expected structure
            return {
                "findings": [
                    {
                        "title": "Missing input validation",
                        "description": "User input should be validated before processing",
                        "severity": "medium",
                        "checklist_category": "Security",
                        "file": "test.py",
                        "line": 10,
                        "confidence": 0.8
                    }
                ]
            }
    
    mock_client = MockBedrockClient()
    reviewer = ReviewerAgent(mock_client)
    
    # Test checklist rules are loaded
    assert "Security" in reviewer.checklist_rules
    assert "Error Handling" in reviewer.checklist_rules
    assert "Architecture & Design Patterns" in reviewer.checklist_rules
    print("✅ Checklist rules loaded correctly")
    
    # Test finding validation
    finding_data = {
        "title": "Test finding",
        "description": "Test description",
        "severity": "high",
        "checklist_category": "Security",
        "file": "test.py",
        "line": 5,
        "confidence": 0.9
    }
    
    test_task = ReviewTask(
        file='test.py',
        depth='deep',
        checklist_sections=['Security'],
        reason='Test task'
    )
    
    validated_finding = reviewer._validate_finding(finding_data, test_task)
    assert validated_finding is not None
    assert validated_finding.title == "Test finding"
    assert validated_finding.severity == "high"
    print("✅ Finding validation works correctly")
    
    # Test file content retrieval
    test_files = [
        ChangedFile(
            path='test.py',
            extension='.py',
            change_type='added',
            lines_added=20,
            lines_removed=0,
            content='def test_function():\n    return "hello"'
        )
    ]
    
    content = reviewer._get_file_content('test.py', test_files)
    assert content == 'def test_function():\n    return "hello"'
    print("✅ File content retrieval works correctly")
    
    return reviewer

def test_reviewer_integration():
    """Test Reviewer Agent integration with mock data"""
    print("\nTesting Reviewer Agent integration...")
    
    # Create mock bedrock client
    class MockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
            return {
                "findings": [
                    {
                        "title": "Missing error handling",
                        "description": "Function should handle potential exceptions",
                        "severity": "medium",
                        "checklist_category": "Error Handling",
                        "file": "test.py",
                        "line": 2,
                        "confidence": 0.7
                    }
                ]
            }
    
    mock_client = MockBedrockClient()
    reviewer = ReviewerAgent(mock_client)
    
    # Create test data
    test_tasks = [
        ReviewTask(
            file='test.py',
            depth='medium',
            checklist_sections=['Error Handling', 'Security'],
            reason='Test file needs review'
        )
    ]
    
    test_files = [
        ChangedFile(
            path='test.py',
            extension='.py',
            change_type='modified',
            lines_added=10,
            lines_removed=2,
            content='def risky_function():\n    return process_user_input()'
        )
    ]
    
    # Test review_code method
    result = reviewer.review_code(test_tasks, test_files)
    
    assert isinstance(result, ReviewerOutput)
    assert len(result.findings) == 1
    assert result.findings[0].title == "Missing error handling"
    assert result.findings[0].severity == "medium"
    print("✅ Reviewer Agent integration test passed")
    
    return result

def main():
    """Run all tests"""
    print("🧪 Testing Reviewer Agent Implementation")
    print("=" * 50)
    
    try:
        # Test data models
        test_finding, test_output = test_reviewer_data_models()
        
        # Test reviewer logic
        reviewer = test_reviewer_logic()
        
        # Test integration
        result = test_reviewer_integration()
        
        print("\n" + "=" * 50)
        print("✅ All Reviewer Agent tests passed!")
        print("🎉 Reviewer Agent implementation is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()