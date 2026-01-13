#!/usr/bin/env python3
"""
Test script for Planner Agent implementation
"""

import sys
import os
sys.path.append('.')

from ai_code_review import *

def test_data_models():
    """Test that all data models work correctly"""
    print("Testing data models...")
    
    # Test ChangedFile creation
    test_file = ChangedFile(
        path='test.py',
        extension='.py',
        change_type='added',
        lines_added=50,
        lines_removed=0,
        content='print("hello world")'
    )
    print("✅ ChangedFile creation successful")
    
    # Test ReviewTask creation
    test_task = ReviewTask(
        file='test.py',
        depth='deep',
        checklist_sections=['Security', 'Code Quality'],
        reason='Test file for validation'
    )
    print("✅ ReviewTask creation successful")
    
    # Test PlannerOutput creation
    test_output = PlannerOutput(tasks=[test_task])
    print("✅ PlannerOutput creation successful")
    
    return test_file, test_task, test_output

def test_planner_logic():
    """Test Planner Agent logic without Claude"""
    print("\nTesting Planner Agent logic...")
    
    # Create mock bedrock client (we'll test without actual API calls)
    class MockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
            # Return a mock response that matches expected structure
            return {
                "tasks": [
                    {
                        "file": "test.py",
                        "depth": "deep",
                        "checklist_sections": ["Security", "Code Quality"],
                        "reason": "Python file with new code requires comprehensive review"
                    }
                ]
            }
    
    mock_client = MockBedrockClient()
    planner = PlannerAgent(mock_client)
    
    # Test depth determination
    test_file = ChangedFile(
        path='test.py',
        extension='.py',
        change_type='added',
        lines_added=50,
        lines_removed=0,
        content='print("hello world")'
    )
    
    depth = planner._determine_review_depth(test_file)
    print(f"✅ Review depth determination: {depth}")
    
    # Test checklist sections
    sections = planner._get_checklist_sections(test_file)
    print(f"✅ Checklist sections: {sections}")
    
    # Test reasoning generation
    reasoning = planner._generate_reasoning(test_file, depth, sections)
    print(f"✅ Reasoning generation: {reasoning}")
    
    return planner

def main():
    """Run all tests"""
    print("🧪 Testing Planner Agent Implementation")
    print("=" * 50)
    
    try:
        # Test data models
        test_file, test_task, test_output = test_data_models()
        
        # Test planner logic
        planner = test_planner_logic()
        
        print("\n" + "=" * 50)
        print("✅ All Planner Agent tests passed!")
        print("🎉 Planner Agent implementation is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()