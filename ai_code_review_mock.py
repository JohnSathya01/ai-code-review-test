#!/usr/bin/env python3
"""
AI Code Review Agent - Mock Version for Testing
This version uses mock responses instead of AWS Bedrock for testing purposes
"""

import os
import sys
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import requests

# Import all the data models and utilities from the main file
# We'll override just the Bedrock client
sys.path.append('.')
from ai_code_review import *

class MockBedrockClient:
    """Mock Bedrock client that returns realistic responses for testing"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        logger.info("Mock Bedrock client initialized for testing")
    
    def invoke_claude(self, prompt: str, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """Mock Claude invocation with realistic responses based on prompt analysis"""
        
        # Analyze the prompt to determine what type of response to return
        prompt_lower = prompt.lower()
        
        if "review plan" in prompt_lower or '"tasks"' in prompt:
            # Planner Agent response
            return {
                "tasks": [
                    {
                        "file": "test_security.py",
                        "depth": "deep",
                        "checklist_sections": ["Security", "Error Handling", "Code Quality"],
                        "reason": "Python file with database operations requires comprehensive security review"
                    },
                    {
                        "file": "file_handler.py", 
                        "depth": "deep",
                        "checklist_sections": ["Security", "Error Handling", "Performance"],
                        "reason": "File handling operations require security and error handling review"
                    }
                ]
            }
        
        elif '"findings"' in prompt:
            # Reviewer Agent response - analyze the file content in the prompt
            findings = []
            
            if "test_security.py" in prompt:
                findings.extend([
                    {
                        "title": "SQL Injection Vulnerability",
                        "description": "User input is directly concatenated into SQL query without sanitization. This creates a critical security vulnerability allowing attackers to execute arbitrary SQL commands. Use parameterized queries instead.",
                        "severity": "high",
                        "checklist_category": "Security",
                        "file": "test_security.py",
                        "line": 3,
                        "confidence": 0.95
                    },
                    {
                        "title": "Missing Input Validation",
                        "description": "Username and password parameters are not validated before use, which could lead to various security issues.",
                        "severity": "medium",
                        "checklist_category": "Security", 
                        "file": "test_security.py",
                        "line": 2,
                        "confidence": 0.80
                    }
                ])
            
            if "file_handler.py" in prompt:
                findings.extend([
                    {
                        "title": "Path Traversal Vulnerability",
                        "description": "Filename parameter is directly concatenated to file path without validation, allowing directory traversal attacks (../../../etc/passwd). Validate and sanitize file paths.",
                        "severity": "high",
                        "checklist_category": "Security",
                        "file": "file_handler.py",
                        "line": 3,
                        "confidence": 0.90
                    },
                    {
                        "title": "Missing Exception Handling",
                        "description": "File operations lack proper exception handling for FileNotFoundError, PermissionError, etc. This could lead to application crashes.",
                        "severity": "medium",
                        "checklist_category": "Error Handling",
                        "file": "file_handler.py",
                        "line": 4,
                        "confidence": 0.85
                    },
                    {
                        "title": "Resource Leak Risk",
                        "description": "File handle may not be properly closed if an exception occurs during reading. Use try-finally or context manager.",
                        "severity": "low",
                        "checklist_category": "Performance",
                        "file": "file_handler.py",
                        "line": 4,
                        "confidence": 0.70
                    }
                ])
            
            # Look for other common patterns
            if "password" in prompt and "admin123" in prompt:
                findings.append({
                    "title": "Hardcoded Credentials",
                    "description": "Password is hardcoded in source code, violating security best practices. Use environment variables or secure configuration.",
                    "severity": "high", 
                    "checklist_category": "Security",
                    "file": "example_code.py",
                    "line": 32,
                    "confidence": 0.95
                })
            
            if "md5" in prompt:
                findings.append({
                    "title": "Weak Cryptographic Hash Function",
                    "description": "MD5 is cryptographically weak and vulnerable to collision attacks. Use SHA-256 or stronger hash functions.",
                    "severity": "medium",
                    "checklist_category": "Security", 
                    "file": "example_code.py",
                    "line": 50,
                    "confidence": 0.80
                })
            
            return {"findings": findings}
        
        elif '"final_issues"' in prompt:
            # Verifier Agent response - keep most high-confidence findings
            issues_data = json.loads(prompt.split('FINDINGS TO VERIFY:')[1].split('Your task is to verify')[0])
            
            final_issues = []
            for i, finding in enumerate(issues_data):
                # Keep high and medium severity findings with good confidence
                if finding.get('severity') == 'high' and finding.get('confidence', 0) >= 0.8:
                    final_issues.append({
                        "id": i,
                        "keep": True,
                        "confidence": min(0.95, finding.get('confidence', 0.8) + 0.05),
                        "severity": "high",
                        "reasoning": "Clear security vulnerability with high confidence"
                    })
                elif finding.get('severity') == 'medium' and finding.get('confidence', 0) >= 0.7:
                    final_issues.append({
                        "id": i, 
                        "keep": True,
                        "confidence": finding.get('confidence', 0.8),
                        "severity": "medium",
                        "reasoning": "Valid issue with sufficient confidence"
                    })
                elif finding.get('severity') == 'low' and finding.get('confidence', 0) >= 0.6:
                    final_issues.append({
                        "id": i,
                        "keep": True, 
                        "confidence": finding.get('confidence', 0.7),
                        "severity": "low",
                        "reasoning": "Minor issue worth addressing"
                    })
            
            return {"final_issues": final_issues}
        
        # Default response
        return {"error": "Unknown prompt type in mock"}

def create_bedrock_client(config: Dict[str, str]) -> MockBedrockClient:
    """Create mock Bedrock client for testing"""
    return MockBedrockClient(config)

# Override the main execution to use mock client
if __name__ == "__main__":
    logger.info(f"AI Code Review Agent v{VERSION} - MOCK VERSION for testing")
    
    # Check if we should use mock mode
    use_mock = os.environ.get('USE_MOCK_AI', 'false').lower() == 'true'
    
    if use_mock:
        logger.info("Using mock AI responses for testing")
        # Set mock AWS credentials for testing
        os.environ.setdefault('AWS_ACCESS_KEY_ID', 'mock_key')
        os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'mock_secret')
        os.environ.setdefault('AWS_REGION', 'us-east-1')
    
    try:
        config = validate_startup_environment()
        logger.info("Environment configuration validated successfully")
        
        # Execute main code review process
        main_execution()
        
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Unexpected startup error: {str(e)}")
        sys.exit(1)