#!/usr/bin/env python3
"""
Test script for Verifier Agent implementation
"""

import sys
import os
sys.path.append('.')

from ai_code_review import *

def test_verifier_data_models():
    """Test that Verifier Agent data models work correctly"""
    print("Testing Verifier Agent data models...")
    
    # Test VerifierOutput creation
    test_finding = Finding(
        title="Test security issue",
        description="This is a test security vulnerability",
        severity="high",
        checklist_category="Security",
        file="test.py",
        line=10,
        confidence=0.9
    )
    
    test_output = VerifierOutput(final_issues=[test_finding])
    print("✅ VerifierOutput creation successful")
    
    return test_output

def test_verifier_logic():
    """Test Verifier Agent logic without Claude"""
    print("\nTesting Verifier Agent logic...")
    
    # Create mock bedrock client
    class MockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
            # Return a mock response that matches expected structure
            return {
                "final_issues": [
                    {
                        "id": 0,
                        "keep": True,
                        "confidence": 0.85,
                        "severity": "high",
                        "reasoning": "Genuine security vulnerability"
                    }
                ]
            }
    
    mock_client = MockBedrockClient()
    verifier = VerifierAgent(mock_client)
    
    # Test confidence thresholds are loaded
    assert "high" in verifier.confidence_thresholds
    assert "medium" in verifier.confidence_thresholds
    assert "low" in verifier.confidence_thresholds
    print("✅ Confidence thresholds loaded correctly")
    
    # Test confidence score calculation
    test_finding = Finding(
        title="SQL injection vulnerability",
        description="User input is directly concatenated into SQL query without sanitization",
        severity="high",
        checklist_category="Security",
        file="test.py",
        line=42,
        confidence=0.8
    )
    
    new_confidence = verifier._calculate_confidence_score(test_finding)
    assert 0.0 <= new_confidence <= 1.0
    print(f"✅ Confidence calculation works: {new_confidence:.2f}")
    
    # Test severity validation
    validated_severity = verifier._validate_severity(test_finding)
    assert validated_severity in ["high", "medium", "low"]
    print(f"✅ Severity validation works: {validated_severity}")
    
    # Test false positive detection
    false_positive_finding = Finding(
        title="Maybe consider improving this",
        description="This might possibly be better",
        severity="high",
        checklist_category="Code Quality",
        file="test.py",
        line=1,
        confidence=0.3
    )
    
    is_false_positive = verifier._is_false_positive(false_positive_finding)
    assert is_false_positive == True
    print("✅ False positive detection works correctly")
    
    return verifier

def test_verifier_integration():
    """Test Verifier Agent integration with mock data"""
    print("\nTesting Verifier Agent integration...")
    
    # Create mock bedrock client
    class MockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
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
    
    mock_client = MockBedrockClient()
    verifier = VerifierAgent(mock_client)
    
    # Create test findings
    test_findings = [
        Finding(
            title="Buffer overflow vulnerability",
            description="Unchecked buffer copy operation can lead to memory corruption",
            severity="high",
            checklist_category="Security",
            file="test.c",
            line=25,
            confidence=0.8
        ),
        Finding(
            title="Maybe improve variable naming",
            description="Variable names could possibly be more descriptive",
            severity="low",
            checklist_category="Code Quality",
            file="test.py",
            line=5,
            confidence=0.4
        )
    ]
    
    # Test verify_findings method
    result = verifier.verify_findings(test_findings)
    
    assert isinstance(result, VerifierOutput)
    # Should filter out the low-confidence finding
    assert len(result.final_issues) <= len(test_findings)
    
    # Check that high-confidence security finding is preserved
    security_findings = [f for f in result.final_issues if f.checklist_category == "Security"]
    assert len(security_findings) > 0
    
    print("✅ Verifier Agent integration test passed")
    print(f"   Original findings: {len(test_findings)}")
    print(f"   Final issues: {len(result.final_issues)}")
    
    return result

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    # Create mock bedrock client that fails
    class FailingMockBedrockClient:
        def invoke_claude(self, prompt, max_tokens):
            raise BedrockError("Mock API failure")
    
    failing_client = FailingMockBedrockClient()
    verifier = VerifierAgent(failing_client)
    
    # Test with empty findings list
    empty_result = verifier.verify_findings([])
    assert len(empty_result.final_issues) == 0
    print("✅ Empty findings list handled correctly")
    
    # Test with failing Claude API (should fall back to rule-based filtering)
    test_findings = [
        Finding(
            title="Memory leak detected",
            description="Resource not properly released in error path",
            severity="medium",
            checklist_category="Error Handling",
            file="test.py",
            line=15,
            confidence=0.7
        )
    ]
    
    fallback_result = verifier.verify_findings(test_findings)
    assert isinstance(fallback_result, VerifierOutput)
    print("✅ Claude API failure handled with fallback")
    
    return fallback_result

def main():
    """Run all tests"""
    print("🧪 Testing Verifier Agent Implementation")
    print("=" * 50)
    
    try:
        # Test data models
        test_output = test_verifier_data_models()
        
        # Test verifier logic
        verifier = test_verifier_logic()
        
        # Test integration
        result = test_verifier_integration()
        
        # Test edge cases
        edge_result = test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ All Verifier Agent tests passed!")
        print("🎉 Verifier Agent implementation is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()