#!/usr/bin/env python3
"""
GitHub Actions Python AI Code Review Agent
A production-grade autonomous code review system using AWS Bedrock Claude models.

This agent implements a three-agent architecture:
- Planner Agent: Determines review strategy
- Reviewer Agent: Performs detailed code analysis  
- Verifier Agent: Filters findings and ensures quality

Author: AI Code Review Agent
Version: 1.0.0
"""

import os
import sys
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import boto3
import requests
from datetime import datetime, timezone

# Version and constants
VERSION = "1.0.0"
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MAX_FILE_SIZE = 1024 * 1024  # 1MB
MAX_FILES_COUNT = 50
MAX_TOKENS = 4000
RETRY_ATTEMPTS = 3

# Configure logging with security filters
class SecurityFilter(logging.Filter):
    """Filter to prevent logging of sensitive information"""
    
    SENSITIVE_PATTERNS = [
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'GITHUB_TOKEN',
        'password', 'secret', 'key', 'token', 'credential'
    ]
    
    def filter(self, record):
        # Never log source code content or credentials
        message = str(record.getMessage()).lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.lower() in message:
                return False
        return True

def setup_logging():
    """Configure structured JSON logging with security filters"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler with JSON formatter
    handler = logging.StreamHandler()
    handler.addFilter(SecurityFilter())
    
    # JSON formatter for structured logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            return json.dumps(log_entry)
    
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    return logger

# Initialize logger
logger = setup_logging()

# Data Models
@dataclass
class ChangedFile:
    """Represents a file that was changed in the git diff"""
    path: str
    extension: str
    change_type: str  # added, modified, deleted
    lines_added: int
    lines_removed: int
    content: str
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.path:
            raise ValueError("File path cannot be empty")
        if self.change_type not in ['added', 'modified', 'deleted']:
            raise ValueError(f"Invalid change_type: {self.change_type}")
        if self.lines_added < 0 or self.lines_removed < 0:
            raise ValueError("Line counts cannot be negative")

@dataclass
class ReviewTask:
    """Represents a code review task assigned by the Planner Agent"""
    file: str
    depth: str  # shallow, medium, deep
    checklist_sections: List[str]
    reason: str
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.file:
            raise ValueError("File path cannot be empty")
        if self.depth not in ['shallow', 'medium', 'deep']:
            raise ValueError(f"Invalid depth: {self.depth}")
        if not self.checklist_sections:
            raise ValueError("Checklist sections cannot be empty")
        if not self.reason:
            raise ValueError("Reason cannot be empty")

@dataclass
class Finding:
    """Represents a code review finding from the Reviewer Agent"""
    title: str
    description: str
    severity: str  # high, medium, low
    checklist_category: str
    file: str
    line: int
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.title:
            raise ValueError("Title cannot be empty")
        if not self.description:
            raise ValueError("Description cannot be empty")
        if self.severity not in ['high', 'medium', 'low']:
            raise ValueError(f"Invalid severity: {self.severity}")
        if not self.checklist_category:
            raise ValueError("Checklist category cannot be empty")
        if not self.file:
            raise ValueError("File path cannot be empty")
        if self.line < 0:
            raise ValueError("Line number cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

@dataclass
class PlannerOutput:
    """Output from the Planner Agent"""
    tasks: List[ReviewTask]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({"tasks": [asdict(task) for task in self.tasks]}, indent=2)

@dataclass
class ReviewerOutput:
    """Output from the Reviewer Agent"""
    findings: List[Finding]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({"findings": [asdict(finding) for finding in self.findings]}, indent=2)

@dataclass
class VerifierOutput:
    """Output from the Verifier Agent"""
    final_issues: List[Finding]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({"final_issues": [asdict(issue) for issue in self.final_issues]}, indent=2)

# Code Review Checklist Categories
CHECKLIST_SECTIONS = [
    "Security",
    "Error Handling", 
    "Architecture & Design Patterns",
    "Performance",
    "Code Quality",
    "Testing",
    "Documentation",
    "Maintainability"
]

# File extension to review depth mapping
EXTENSION_DEPTH_MAP = {
    '.py': 'deep',
    '.js': 'deep', 
    '.ts': 'deep',
    '.java': 'deep',
    '.cpp': 'deep',
    '.c': 'deep',
    '.go': 'medium',
    '.rs': 'medium',
    '.php': 'medium',
    '.rb': 'medium',
    '.sql': 'medium',
    '.yaml': 'shallow',
    '.yml': 'shallow',
    '.json': 'shallow',
    '.md': 'shallow',
    '.txt': 'shallow'
}

# File extension to checklist sections mapping
EXTENSION_CHECKLIST_MAP = {
    '.py': ["Security", "Error Handling", "Architecture & Design Patterns", "Code Quality"],
    '.js': ["Security", "Error Handling", "Performance", "Code Quality"],
    '.ts': ["Security", "Error Handling", "Performance", "Code Quality"],
    '.java': ["Security", "Error Handling", "Architecture & Design Patterns", "Performance"],
    '.cpp': ["Security", "Error Handling", "Performance", "Maintainability"],
    '.c': ["Security", "Error Handling", "Performance", "Maintainability"],
    '.go': ["Security", "Error Handling", "Performance"],
    '.rs': ["Security", "Error Handling", "Performance"],
    '.php': ["Security", "Error Handling", "Code Quality"],
    '.rb': ["Security", "Error Handling", "Code Quality"],
    '.sql': ["Security", "Performance"],
    '.yaml': ["Security", "Code Quality"],
    '.yml': ["Security", "Code Quality"],
    '.json': ["Security"],
    '.md': ["Documentation"],
    '.txt': ["Documentation"]
}

# Required environment variables
REQUIRED_ENV_VARS = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY', 
    'AWS_REGION',
    'GITHUB_TOKEN',
    'GITHUB_ACTOR',
    'GITHUB_REPOSITORY'
]

# Environment Configuration and Validation
class EnvironmentError(Exception):
    """Custom exception for environment configuration errors"""
    pass

def validate_environment_variables() -> Dict[str, str]:
    """
    Validate that all required environment variables are present and non-empty.
    
    Returns:
        Dict[str, str]: Dictionary of validated environment variables
        
    Raises:
        EnvironmentError: If any required environment variable is missing or empty
    """
    logger.info("Validating environment configuration...")
    
    env_vars = {}
    missing_vars = []
    empty_vars = []
    
    for var_name in REQUIRED_ENV_VARS:
        value = os.environ.get(var_name)
        
        if value is None:
            missing_vars.append(var_name)
        elif not value.strip():
            empty_vars.append(var_name)
        else:
            # Store the variable but never log its value for security
            env_vars[var_name] = value.strip()
    
    # Report all missing variables at once for better user experience
    if missing_vars or empty_vars:
        error_parts = []
        
        if missing_vars:
            error_parts.append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        if empty_vars:
            error_parts.append(f"Empty environment variables: {', '.join(empty_vars)}")
        
        error_message = ". ".join(error_parts)
        error_message += ". Please ensure all required environment variables are set with valid values."
        
        logger.error(f"Environment validation failed: {error_message}")
        raise EnvironmentError(error_message)
    
    logger.info(f"Environment validation successful - all {len(REQUIRED_ENV_VARS)} required variables present")
    return env_vars

def get_secure_config() -> Dict[str, str]:
    """
    Get validated environment configuration with secure credential handling.
    
    This function ensures:
    - All required environment variables are present
    - No credentials are hardcoded
    - No sensitive values are logged
    - Clear error messages for missing configuration
    
    Returns:
        Dict[str, str]: Validated configuration dictionary
        
    Raises:
        EnvironmentError: If environment validation fails
    """
    try:
        config = validate_environment_variables()
        
        # Additional validation for specific variables
        aws_region = config.get('AWS_REGION', '').lower()
        if aws_region and not aws_region.startswith(('us-', 'eu-', 'ap-', 'ca-', 'sa-', 'af-', 'me-')):
            logger.warning(f"AWS region '{aws_region}' may not be valid - proceeding anyway")
        
        github_repo = config.get('GITHUB_REPOSITORY', '')
        if github_repo and '/' not in github_repo:
            logger.warning(f"GITHUB_REPOSITORY '{github_repo}' should be in format 'owner/repo'")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get secure configuration: {str(e)}")
        raise

def validate_startup_environment() -> Dict[str, str]:
    """
    Perform comprehensive startup environment validation.
    
    This function is called at application startup to ensure all
    required configuration is present before proceeding with execution.
    
    Returns:
        Dict[str, str]: Validated environment configuration
        
    Raises:
        EnvironmentError: If any validation fails
        SystemExit: If critical configuration is missing
    """
    try:
        logger.info("Starting environment validation...")
        
        # Validate all required environment variables
        config = get_secure_config()
        
        # Log successful validation without exposing sensitive values
        logger.info("Environment validation completed successfully")
        logger.info(f"Configured for AWS region: {config['AWS_REGION']}")
        logger.info(f"Configured for GitHub repository: {config['GITHUB_REPOSITORY']}")
        logger.info(f"Configured for GitHub actor: {config['GITHUB_ACTOR']}")
        
        return config
        
    except EnvironmentError as e:
        logger.error(f"Environment validation failed: {str(e)}")
        logger.error("Application cannot start without proper configuration")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error during environment validation: {str(e)}")
        logger.error("Application startup failed")
        sys.exit(1)


# Git Integration Module
class GitError(Exception):
    """Custom exception for Git-related errors"""
    pass


def get_git_diff() -> str:
    """
    Get the git diff for changed files.
    
    Returns:
        str: Raw git diff output
        
    Raises:
        GitError: If git diff command fails
    """
    try:
        logger.info("Fetching git diff...")
        
        # First, check if we have any commits
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            # No commits yet, get all staged/unstaged files
            logger.info("No commits found, checking for staged and unstaged files...")
            
            # Get staged files
            result = subprocess.run(
                ['git', 'diff', '--name-status', '--cached'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            staged_output = result.stdout.strip() if result.returncode == 0 else ""
            
            # Get unstaged files
            result = subprocess.run(
                ['git', 'diff', '--name-status'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            unstaged_output = result.stdout.strip() if result.returncode == 0 else ""
            
            # Combine outputs
            diff_output = ""
            if staged_output:
                diff_output += staged_output
            if unstaged_output:
                if diff_output:
                    diff_output += "\n"
                diff_output += unstaged_output
            
            # If no staged/unstaged files, get all tracked files as "added"
            if not diff_output:
                result = subprocess.run(
                    ['git', 'ls-files'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    # Format as added files
                    files = result.stdout.strip().split('\n')
                    diff_output = '\n'.join([f"A\t{f}" for f in files if f.strip()])
            
        else:
            # We have commits, try different diff strategies
            # Try to get diff against previous commit
            result = subprocess.run(
                ['git', 'diff', '--name-status', 'HEAD^', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                # Try alternative: diff against HEAD~1
                result = subprocess.run(
                    ['git', 'diff', '--name-status', 'HEAD~1'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    # Try diff against origin/main or origin/master
                    for branch in ['origin/main', 'origin/master']:
                        result = subprocess.run(
                            ['git', 'diff', '--name-status', branch],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            break
                    
                    if result.returncode != 0:
                        raise GitError(f"Git diff command failed: {result.stderr}")
            
            diff_output = result.stdout.strip()
        
        if not diff_output:
            logger.warning("No changes detected in git diff")
            return ""
        
        logger.info(f"Git diff retrieved successfully")
        return diff_output
        
    except subprocess.TimeoutExpired:
        raise GitError("Git diff command timed out after 30 seconds")
    except FileNotFoundError:
        raise GitError("Git command not found - ensure git is installed")
    except Exception as e:
        raise GitError(f"Failed to get git diff: {str(e)}")


def parse_git_diff_line(line: str) -> Optional[tuple]:
    """
    Parse a single line from git diff --name-status output.
    
    Args:
        line: A line from git diff output (e.g., "M\tfile.py")
        
    Returns:
        Optional[tuple]: (change_type, file_path) or None if invalid
    """
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None
    
    status = parts[0].upper()
    file_path = parts[1]
    
    # Map git status codes to our change types
    change_type_map = {
        'A': 'added',
        'M': 'modified',
        'D': 'deleted',
        'R': 'modified',  # Renamed files treated as modified
        'C': 'added',     # Copied files treated as added
        'T': 'modified'   # Type changed treated as modified
    }
    
    change_type = change_type_map.get(status[0], 'modified')
    
    return (change_type, file_path)


def get_file_stats(file_path: str) -> tuple:
    """
    Get line addition and deletion statistics for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        tuple: (lines_added, lines_removed)
    """
    try:
        # Check if we have commits
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            # No commits, count lines in file as added
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                    return (lines, 0)
                except Exception:
                    return (0, 0)
            return (0, 0)
        
        # We have commits, get diff stats
        result = subprocess.run(
            ['git', 'diff', '--numstat', 'HEAD^', 'HEAD', '--', file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split('\t')
            if len(parts) >= 2:
                lines_added = int(parts[0]) if parts[0] != '-' else 0
                lines_removed = int(parts[1]) if parts[1] != '-' else 0
                return (lines_added, lines_removed)
        
        # Try alternative diff if previous failed
        result = subprocess.run(
            ['git', 'diff', '--numstat', 'HEAD~1', '--', file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split('\t')
            if len(parts) >= 2:
                lines_added = int(parts[0]) if parts[0] != '-' else 0
                lines_removed = int(parts[1]) if parts[1] != '-' else 0
                return (lines_added, lines_removed)
        
        return (0, 0)
        
    except Exception as e:
        logger.warning(f"Failed to get file stats for {file_path}: {str(e)}")
        return (0, 0)


def read_file_content(file_path: str, max_size: int = MAX_FILE_SIZE) -> str:
    """
    Read file content with size limit enforcement.
    
    Args:
        file_path: Path to the file
        max_size: Maximum file size in bytes
        
    Returns:
        str: File content or empty string if file doesn't exist/is too large
        
    Raises:
        GitError: If file exceeds size limit
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            raise GitError(
                f"File {file_path} exceeds size limit: {file_size} bytes > {max_size} bytes"
            )
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return content
        
    except GitError:
        raise
    except Exception as e:
        logger.warning(f"Failed to read file {file_path}: {str(e)}")
        return ""


def get_changed_files() -> List[ChangedFile]:
    """
    Extract changed files from git diff with resource limit enforcement.
    
    This function:
    - Gets git diff output
    - Parses file paths, change types, and line counts
    - Reads file content with error handling
    - Enforces file size and count limits
    
    Returns:
        List[ChangedFile]: List of changed files with metadata
        
    Raises:
        GitError: If git operations fail or limits are exceeded
    """
    try:
        logger.info("Extracting changed files from git diff...")
        
        # Get git diff
        diff_output = get_git_diff()
        
        if not diff_output:
            logger.info("No changed files found")
            return []
        
        # Parse diff output
        changed_files = []
        lines = diff_output.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            parsed = parse_git_diff_line(line)
            if not parsed:
                logger.warning(f"Failed to parse git diff line: {line}")
                continue
            
            change_type, file_path = parsed
            
            # Get file extension
            extension = os.path.splitext(file_path)[1].lower()
            
            # Get file statistics
            lines_added, lines_removed = get_file_stats(file_path)
            
            # Read file content (skip deleted files)
            content = ""
            if change_type != 'deleted':
                try:
                    content = read_file_content(file_path)
                except GitError as e:
                    logger.warning(f"Skipping file due to size limit: {file_path}")
                    continue
            
            # Create ChangedFile object
            changed_file = ChangedFile(
                path=file_path,
                extension=extension,
                change_type=change_type,
                lines_added=lines_added,
                lines_removed=lines_removed,
                content=content
            )
            
            changed_files.append(changed_file)
            
            # Enforce file count limit
            if len(changed_files) >= MAX_FILES_COUNT:
                logger.warning(
                    f"Reached maximum file count limit ({MAX_FILES_COUNT}). "
                    f"Processing first {MAX_FILES_COUNT} files only."
                )
                break
        
        logger.info(f"Successfully extracted {len(changed_files)} changed files")
        
        # Log file summary without exposing content
        for cf in changed_files:
            logger.info(
                f"Changed file: {cf.path} ({cf.change_type}, "
                f"+{cf.lines_added}/-{cf.lines_removed})"
            )
        
        return changed_files
        
    except GitError:
        raise
    except Exception as e:
        raise GitError(f"Failed to extract changed files: {str(e)}")


# AWS Bedrock Claude Integration Module
class BedrockError(Exception):
    """Custom exception for AWS Bedrock-related errors"""
    pass


class BedrockClient:
    """AWS Bedrock Claude client with authentication and retry logic"""
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize Bedrock client with configuration.
        
        Args:
            config: Dictionary containing AWS credentials and region
        """
        self.config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize boto3 bedrock-runtime client with proper authentication"""
        try:
            logger.info("Initializing AWS Bedrock client...")
            
            # Create bedrock-runtime client with explicit credentials
            self._client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.config['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=self.config['AWS_SECRET_ACCESS_KEY'],
                region_name=self.config['AWS_REGION']
            )
            
            logger.info(f"AWS Bedrock client initialized for region: {self.config['AWS_REGION']}")
            
        except Exception as e:
            raise BedrockError(f"Failed to initialize AWS Bedrock client: {str(e)}")
    
    def _validate_token_limit(self, prompt: str, max_tokens: int) -> None:
        """
        Validate that prompt and response don't exceed token limits.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens for response
            
        Raises:
            BedrockError: If token limits are exceeded
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        estimated_prompt_tokens = len(prompt) // 4
        
        # Claude 3.5 Sonnet has a 200k token context limit
        max_context_tokens = 200000
        
        if estimated_prompt_tokens > max_context_tokens:
            raise BedrockError(
                f"Prompt too long: estimated {estimated_prompt_tokens} tokens "
                f"exceeds context limit of {max_context_tokens} tokens"
            )
        
        if max_tokens > 4096:  # Claude output limit
            raise BedrockError(
                f"Requested max_tokens {max_tokens} exceeds Claude output limit of 4096"
            )
        
        if estimated_prompt_tokens + max_tokens > max_context_tokens:
            raise BedrockError(
                f"Total tokens (prompt + response) would exceed context limit: "
                f"{estimated_prompt_tokens} + {max_tokens} > {max_context_tokens}"
            )
    
    def invoke_claude(self, prompt: str, max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """
        Invoke Claude model with retry logic and JSON response parsing.
        
        Args:
            prompt: Input prompt for Claude
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict[str, Any]: Parsed JSON response from Claude
            
        Raises:
            BedrockError: If API call fails or response is invalid
        """
        # Validate token limits
        self._validate_token_limit(prompt, max_tokens)
        
        # Prepare request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent, focused responses
            "top_p": 0.9
        }
        
        last_error = None
        
        # Retry logic for API calls
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                logger.info(f"Invoking Claude model (attempt {attempt}/{RETRY_ATTEMPTS})...")
                
                # Make API call
                response = self._client.invoke_model(
                    modelId=CLAUDE_MODEL_ID,
                    body=json.dumps(request_body),
                    contentType='application/json',
                    accept='application/json'
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                
                # Validate response structure
                if 'content' not in response_body:
                    raise BedrockError("Invalid response structure: missing 'content' field")
                
                if not response_body['content'] or len(response_body['content']) == 0:
                    raise BedrockError("Empty response content from Claude")
                
                # Extract text content
                content = response_body['content'][0]
                if 'text' not in content:
                    raise BedrockError("Invalid response structure: missing 'text' field")
                
                response_text = content['text'].strip()
                
                if not response_text:
                    raise BedrockError("Empty text response from Claude")
                
                logger.info("Claude model invocation successful")
                
                # Try to parse as JSON
                try:
                    parsed_json = json.loads(response_text)
                    return parsed_json
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, try to extract JSON from response
                    json_match = self._extract_json_from_text(response_text)
                    if json_match:
                        try:
                            parsed_json = json.loads(json_match)
                            logger.info("Successfully extracted JSON from Claude response")
                            return parsed_json
                        except json.JSONDecodeError:
                            pass
                    
                    if attempt == RETRY_ATTEMPTS:
                        raise BedrockError(
                            f"Failed to parse JSON response after {RETRY_ATTEMPTS} attempts. "
                            f"Last JSON error: {str(e)}"
                        )
                    else:
                        logger.warning(f"JSON parsing failed on attempt {attempt}, retrying...")
                        last_error = BedrockError(f"JSON parsing error: {str(e)}")
                        continue
                
            except BedrockError:
                raise
            except Exception as e:
                last_error = BedrockError(f"AWS Bedrock API error: {str(e)}")
                
                if attempt == RETRY_ATTEMPTS:
                    logger.error(f"All {RETRY_ATTEMPTS} attempts failed")
                    raise last_error
                else:
                    logger.warning(f"Attempt {attempt} failed, retrying: {str(e)}")
                    # Exponential backoff
                    import time
                    time.sleep(2 ** (attempt - 1))
        
        # This should never be reached, but just in case
        raise last_error or BedrockError("Unknown error in Claude invocation")
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text response that may contain additional formatting.
        
        Args:
            text: Raw text response from Claude
            
        Returns:
            Optional[str]: Extracted JSON string or None if not found
        """
        import re
        
        # Try to find JSON blocks in various formats
        patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON code blocks
            r'```\s*(\{.*?\})\s*```',      # Generic code blocks
            r'(\{.*?\})',                   # Any JSON-like structure
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Return the largest match (most likely to be complete)
                return max(matches, key=len)
        
        return None


def create_bedrock_client(config: Dict[str, str]) -> BedrockClient:
    """
    Create and validate AWS Bedrock client.
    
    Args:
        config: Environment configuration dictionary
        
    Returns:
        BedrockClient: Initialized Bedrock client
        
    Raises:
        BedrockError: If client creation fails
    """
    try:
        client = BedrockClient(config)
        
        # Test the client with a simple request
        logger.info("Testing AWS Bedrock client connectivity...")
        
        test_response = client.invoke_claude(
            prompt='{"test": "connectivity"}',
            max_tokens=50
        )
        
        logger.info("AWS Bedrock client test successful")
        return client
        
    except Exception as e:
        raise BedrockError(f"Failed to create Bedrock client: {str(e)}")


def validate_api_security(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize API response data for security.
    
    Args:
        response_data: Raw response data from Claude
        
    Returns:
        Dict[str, Any]: Validated and sanitized response data
        
    Raises:
        BedrockError: If response contains security issues
    """
    if not isinstance(response_data, dict):
        raise BedrockError("Response must be a dictionary")
    
    # Check for potentially dangerous content
    dangerous_patterns = [
        'eval(',
        'exec(',
        '__import__',
        'subprocess',
        'os.system',
        'shell=True'
    ]
    
    response_str = json.dumps(response_data).lower()
    
    for pattern in dangerous_patterns:
        if pattern in response_str:
            logger.warning(f"Potentially dangerous content detected: {pattern}")
            # Don't fail, but log for security monitoring
    
    # Validate required structure based on response type
    # This will be expanded when we implement the agents
    
    return response_data


# Planner Agent Implementation
class PlannerAgent:
    """
    Planner Agent responsible for determining review strategy for changed files.
    
    The Planner Agent analyzes changed files and determines:
    - Which files need review
    - Review depth for each file (shallow, medium, deep)
    - Applicable checklist sections based on file type and changes
    - Reasoning for decisions
    """
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize Planner Agent with Bedrock client.
        
        Args:
            bedrock_client: AWS Bedrock client for Claude interactions
        """
        self.bedrock_client = bedrock_client
        self.logger = logging.getLogger(__name__)
    
    def _determine_review_depth(self, changed_file: ChangedFile) -> str:
        """
        Determine review depth based on file characteristics.
        
        Args:
            changed_file: File to analyze
            
        Returns:
            str: Review depth ('shallow', 'medium', 'deep')
        """
        # Get base depth from extension mapping
        base_depth = EXTENSION_DEPTH_MAP.get(changed_file.extension, 'medium')
        
        # Adjust depth based on change characteristics
        total_changes = changed_file.lines_added + changed_file.lines_removed
        
        # Large changes get deeper review
        if total_changes > 100:
            if base_depth == 'shallow':
                return 'medium'
            elif base_depth == 'medium':
                return 'deep'
        
        # New files get deeper review
        if changed_file.change_type == 'added' and total_changes > 20:
            if base_depth == 'shallow':
                return 'medium'
        
        # Deleted files get shallow review
        if changed_file.change_type == 'deleted':
            return 'shallow'
        
        return base_depth
    
    def _get_checklist_sections(self, changed_file: ChangedFile) -> List[str]:
        """
        Get applicable checklist sections for a file.
        
        Args:
            changed_file: File to analyze
            
        Returns:
            List[str]: List of applicable checklist sections
        """
        # Get base sections from extension mapping
        base_sections = EXTENSION_CHECKLIST_MAP.get(changed_file.extension, ["Code Quality"])
        
        # Add additional sections based on file characteristics
        sections = base_sections.copy()
        
        # Large changes get additional scrutiny
        total_changes = changed_file.lines_added + changed_file.lines_removed
        if total_changes > 50:
            if "Testing" not in sections:
                sections.append("Testing")
            if "Maintainability" not in sections:
                sections.append("Maintainability")
        
        # New files get comprehensive review
        if changed_file.change_type == 'added':
            if "Documentation" not in sections:
                sections.append("Documentation")
        
        # Security-sensitive file patterns
        security_patterns = [
            'auth', 'login', 'password', 'token', 'key', 'secret',
            'crypto', 'hash', 'encrypt', 'decrypt', 'ssl', 'tls'
        ]
        
        file_path_lower = changed_file.path.lower()
        if any(pattern in file_path_lower for pattern in security_patterns):
            if "Security" not in sections:
                sections.append("Security")
        
        return sections
    
    def _generate_reasoning(self, changed_file: ChangedFile, depth: str, sections: List[str]) -> str:
        """
        Generate reasoning for review decisions.
        
        Args:
            changed_file: File being analyzed
            depth: Determined review depth
            sections: Applicable checklist sections
            
        Returns:
            str: Human-readable reasoning
        """
        reasons = []
        
        # File type reasoning
        if changed_file.extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
            reasons.append(f"Core programming language file ({changed_file.extension})")
        elif changed_file.extension in ['.yaml', '.yml', '.json']:
            reasons.append(f"Configuration file ({changed_file.extension})")
        elif changed_file.extension in ['.md', '.txt']:
            reasons.append(f"Documentation file ({changed_file.extension})")
        
        # Change type reasoning
        if changed_file.change_type == 'added':
            reasons.append("New file requires comprehensive review")
        elif changed_file.change_type == 'deleted':
            reasons.append("Deleted file requires minimal review")
        else:
            reasons.append("Modified file needs change analysis")
        
        # Size reasoning
        total_changes = changed_file.lines_added + changed_file.lines_removed
        if total_changes > 100:
            reasons.append(f"Large change set ({total_changes} lines)")
        elif total_changes > 20:
            reasons.append(f"Moderate change set ({total_changes} lines)")
        else:
            reasons.append(f"Small change set ({total_changes} lines)")
        
        # Depth reasoning
        if depth == 'deep':
            reasons.append("Deep review for critical code paths")
        elif depth == 'medium':
            reasons.append("Medium review for standard code changes")
        else:
            reasons.append("Shallow review for low-risk changes")
        
        return "; ".join(reasons)
    
    def _create_claude_prompt(self, changed_files: List[ChangedFile]) -> str:
        """
        Create prompt for Claude to analyze files and generate review tasks.
        
        Args:
            changed_files: List of changed files to analyze
            
        Returns:
            str: Formatted prompt for Claude
        """
        # Create file summaries without exposing source code content
        file_summaries = []
        for cf in changed_files:
            summary = {
                "path": cf.path,
                "extension": cf.extension,
                "change_type": cf.change_type,
                "lines_added": cf.lines_added,
                "lines_removed": cf.lines_removed,
                "content_length": len(cf.content) if cf.content else 0
            }
            file_summaries.append(summary)
        
        prompt = f"""You are a code review planning agent. Analyze the following changed files and create a review plan.

CHANGED FILES:
{json.dumps(file_summaries, indent=2)}

AVAILABLE CHECKLIST SECTIONS:
{json.dumps(CHECKLIST_SECTIONS, indent=2)}

REVIEW DEPTH OPTIONS:
- shallow: Basic syntax and obvious issues
- medium: Standard code review practices
- deep: Comprehensive analysis including architecture and security

Your task is to create a structured review plan. For each file that needs review, determine:
1. Review depth (shallow/medium/deep)
2. Applicable checklist sections
3. Clear reasoning for decisions

Focus on:
- Security-sensitive files need deeper review
- Large changes need more comprehensive analysis
- File types determine applicable checklist sections
- New files need thorough review
- Configuration files need security focus

Return ONLY a valid JSON object with this exact structure:
{{
  "tasks": [
    {{
      "file": "path/to/file.py",
      "depth": "deep",
      "checklist_sections": ["Security", "Error Handling"],
      "reason": "New Python file with authentication logic requires comprehensive security review"
    }}
  ]
}}

Do not include any explanation outside the JSON object."""
        
        return prompt
    
    def plan_review(self, changed_files: List[ChangedFile]) -> PlannerOutput:
        """
        Plan review strategy for changed files.
        
        This method analyzes changed files and determines the optimal review strategy
        by combining rule-based logic with Claude's analysis capabilities.
        
        Args:
            changed_files: List of files that have been changed
            
        Returns:
            PlannerOutput: Structured review plan with tasks
            
        Raises:
            BedrockError: If Claude analysis fails
        """
        try:
            self.logger.info(f"Planning review for {len(changed_files)} changed files...")
            
            if not changed_files:
                self.logger.info("No changed files to review")
                return PlannerOutput(tasks=[])
            
            # Filter out files that don't need review (e.g., very large files, binary files)
            reviewable_files = []
            for cf in changed_files:
                # Skip binary files and very large files
                if cf.extension in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.tar', '.gz']:
                    self.logger.info(f"Skipping binary file: {cf.path}")
                    continue
                
                # Skip files that are too large
                if len(cf.content) > MAX_FILE_SIZE:
                    self.logger.warning(f"Skipping large file: {cf.path} ({len(cf.content)} bytes)")
                    continue
                
                reviewable_files.append(cf)
            
            if not reviewable_files:
                self.logger.info("No reviewable files found")
                return PlannerOutput(tasks=[])
            
            # Create initial plan using rule-based logic
            initial_tasks = []
            for cf in reviewable_files:
                depth = self._determine_review_depth(cf)
                sections = self._get_checklist_sections(cf)
                reason = self._generate_reasoning(cf, depth, sections)
                
                task = ReviewTask(
                    file=cf.path,
                    depth=depth,
                    checklist_sections=sections,
                    reason=reason
                )
                initial_tasks.append(task)
            
            # Use Claude to refine the plan
            try:
                prompt = self._create_claude_prompt(reviewable_files)
                claude_response = self.bedrock_client.invoke_claude(prompt, max_tokens=2000)
                
                # Validate Claude's response structure
                if "tasks" not in claude_response:
                    raise BedrockError("Invalid Claude response: missing 'tasks' field")
                
                claude_tasks = claude_response["tasks"]
                if not isinstance(claude_tasks, list):
                    raise BedrockError("Invalid Claude response: 'tasks' must be a list")
                
                # Convert Claude's response to ReviewTask objects
                refined_tasks = []
                for task_data in claude_tasks:
                    try:
                        # Validate required fields
                        required_fields = ["file", "depth", "checklist_sections", "reason"]
                        for field in required_fields:
                            if field not in task_data:
                                raise ValueError(f"Missing required field: {field}")
                        
                        # Validate depth value
                        if task_data["depth"] not in ["shallow", "medium", "deep"]:
                            self.logger.warning(f"Invalid depth '{task_data['depth']}' for {task_data['file']}, using 'medium'")
                            task_data["depth"] = "medium"
                        
                        # Validate checklist sections
                        valid_sections = []
                        for section in task_data["checklist_sections"]:
                            if section in CHECKLIST_SECTIONS:
                                valid_sections.append(section)
                            else:
                                self.logger.warning(f"Invalid checklist section '{section}' ignored")
                        
                        if not valid_sections:
                            valid_sections = ["Code Quality"]  # Default fallback
                        
                        task = ReviewTask(
                            file=task_data["file"],
                            depth=task_data["depth"],
                            checklist_sections=valid_sections,
                            reason=task_data["reason"]
                        )
                        refined_tasks.append(task)
                        
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Invalid task data from Claude: {e}")
                        # Fall back to rule-based task for this file
                        matching_initial = next((t for t in initial_tasks if t.file == task_data.get("file")), None)
                        if matching_initial:
                            refined_tasks.append(matching_initial)
                
                # Use refined tasks if we got valid results, otherwise fall back to initial
                final_tasks = refined_tasks if refined_tasks else initial_tasks
                
            except BedrockError as e:
                self.logger.warning(f"Claude analysis failed, using rule-based plan: {e}")
                final_tasks = initial_tasks
            
            # Log the final plan (without exposing file content)
            self.logger.info(f"Review plan created with {len(final_tasks)} tasks")
            for task in final_tasks:
                self.logger.info(
                    f"Task: {task.file} -> {task.depth} review, "
                    f"sections: {', '.join(task.checklist_sections)}"
                )
            
            return PlannerOutput(tasks=final_tasks)
            
        except Exception as e:
            self.logger.error(f"Failed to plan review: {str(e)}")
            raise BedrockError(f"Planner Agent failed: {str(e)}")


def create_planner_agent(bedrock_client: BedrockClient) -> PlannerAgent:
    """
    Create and initialize Planner Agent.
    
    Args:
        bedrock_client: AWS Bedrock client for Claude interactions
        
    Returns:
        PlannerAgent: Initialized Planner Agent
    """
    return PlannerAgent(bedrock_client)


# Reviewer Agent Implementation
class ReviewerAgent:
    """
    Reviewer Agent responsible for performing detailed code analysis.
    
    The Reviewer Agent analyzes code files and:
    - Applies enterprise code review checklist as hard constraints
    - Identifies security, architecture, and error handling issues
    - Categorizes findings by checklist sections
    - Assigns severity levels to findings
    - Generates structured JSON output with detailed findings
    """
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize Reviewer Agent with Bedrock client.
        
        Args:
            bedrock_client: AWS Bedrock client for Claude interactions
        """
        self.bedrock_client = bedrock_client
        self.logger = logging.getLogger(__name__)
        
        # Define the enterprise code review checklist
        self.checklist_rules = {
            "Security": [
                "Input validation and sanitization",
                "Authentication and authorization checks", 
                "Secure credential handling",
                "SQL injection prevention",
                "XSS prevention",
                "CSRF protection",
                "Secure communication (HTTPS/TLS)",
                "Sensitive data exposure prevention",
                "Cryptographic best practices",
                "Access control validation"
            ],
            "Error Handling": [
                "Proper exception handling",
                "Graceful error recovery",
                "Error logging without sensitive data exposure",
                "User-friendly error messages",
                "Resource cleanup in error paths",
                "Timeout handling",
                "Retry logic implementation",
                "Circuit breaker patterns",
                "Fallback mechanisms",
                "Error propagation consistency"
            ],
            "Architecture & Design Patterns": [
                "Single Responsibility Principle",
                "Dependency injection usage",
                "Interface segregation",
                "Proper abstraction layers",
                "Design pattern implementation",
                "Code modularity and reusability",
                "Separation of concerns",
                "SOLID principles adherence",
                "Clean architecture practices",
                "Coupling and cohesion optimization"
            ],
            "Performance": [
                "Algorithm efficiency",
                "Database query optimization",
                "Memory usage optimization",
                "Caching strategies",
                "Resource pooling",
                "Lazy loading implementation",
                "Batch processing optimization",
                "Network call optimization",
                "Concurrent processing efficiency",
                "Scalability considerations"
            ],
            "Code Quality": [
                "Code readability and clarity",
                "Naming conventions consistency",
                "Function and class size appropriateness",
                "Code duplication elimination",
                "Comment quality and necessity",
                "Magic number elimination",
                "Dead code removal",
                "Consistent formatting",
                "Type safety enforcement",
                "Code complexity management"
            ],
            "Testing": [
                "Unit test coverage",
                "Integration test adequacy",
                "Test case quality",
                "Mock usage appropriateness",
                "Test data management",
                "Test isolation",
                "Edge case coverage",
                "Error condition testing",
                "Performance test considerations",
                "Test maintainability"
            ],
            "Documentation": [
                "API documentation completeness",
                "Code comment accuracy",
                "README file quality",
                "Configuration documentation",
                "Deployment guide completeness",
                "Change log maintenance",
                "Architecture documentation",
                "User guide quality",
                "Troubleshooting documentation",
                "Version compatibility notes"
            ],
            "Maintainability": [
                "Code organization structure",
                "Configuration externalization",
                "Logging implementation",
                "Monitoring and observability",
                "Version control best practices",
                "Dependency management",
                "Build process optimization",
                "Deployment automation",
                "Environment consistency",
                "Technical debt management"
            ]
        }
    
    def _create_checklist_prompt(self, task: ReviewTask, file_content: str) -> str:
        """
        Create a detailed prompt for Claude to review code against specific checklist sections.
        
        Args:
            task: Review task with file and checklist sections
            file_content: Content of the file to review
            
        Returns:
            str: Formatted prompt for Claude
        """
        # Get the specific checklist rules for this task
        applicable_rules = {}
        for section in task.checklist_sections:
            if section in self.checklist_rules:
                applicable_rules[section] = self.checklist_rules[section]
        
        prompt = f"""You are an expert code reviewer performing a {task.depth} review of a {task.file} file.

REVIEW TASK:
- File: {task.file}
- Review Depth: {task.depth}
- Reason: {task.reason}

CHECKLIST SECTIONS TO APPLY:
{json.dumps(applicable_rules, indent=2)}

FILE CONTENT TO REVIEW:
```
{file_content}
```

REVIEW INSTRUCTIONS:
1. Apply the checklist rules as HARD CONSTRAINTS - every rule must be validated
2. Identify specific issues, not general observations
3. Focus on {task.depth} level analysis:
   - shallow: Basic syntax and obvious issues only
   - medium: Standard code review practices
   - deep: Comprehensive analysis including architecture and security

4. For each finding, determine:
   - Specific title describing the issue
   - Detailed description with context
   - Severity level (high/medium/low):
     * high: Security vulnerabilities, critical bugs, major architectural flaws
     * medium: Performance issues, maintainability concerns, minor bugs
     * low: Style issues, minor improvements, documentation gaps
   - Exact line number where the issue occurs
   - Which checklist category it violates

5. Only report genuine issues that violate the checklist rules
6. Be specific about line numbers and code snippets
7. Provide actionable recommendations

Return ONLY a valid JSON object with this exact structure:
{{
  "findings": [
    {{
      "title": "Specific issue title",
      "description": "Detailed description with context and recommendation",
      "severity": "high|medium|low",
      "checklist_category": "Security|Error Handling|Architecture & Design Patterns|Performance|Code Quality|Testing|Documentation|Maintainability",
      "file": "{task.file}",
      "line": 42,
      "confidence": 0.9
    }}
  ]
}}

If no issues are found, return: {{"findings": []}}

Do not include any explanation outside the JSON object."""
        
        return prompt
    
    def _validate_finding(self, finding_data: Dict[str, Any], task: ReviewTask) -> Optional[Finding]:
        """
        Validate and create a Finding object from Claude's response data.
        
        Args:
            finding_data: Raw finding data from Claude
            task: Review task context
            
        Returns:
            Optional[Finding]: Validated Finding object or None if invalid
        """
        try:
            # Validate required fields
            required_fields = ["title", "description", "severity", "checklist_category", "file", "line"]
            for field in required_fields:
                if field not in finding_data:
                    self.logger.warning(f"Missing required field '{field}' in finding")
                    return None
            
            # Validate severity
            if finding_data["severity"] not in ["high", "medium", "low"]:
                self.logger.warning(f"Invalid severity '{finding_data['severity']}', defaulting to 'medium'")
                finding_data["severity"] = "medium"
            
            # Validate checklist category
            if finding_data["checklist_category"] not in CHECKLIST_SECTIONS:
                self.logger.warning(f"Invalid checklist category '{finding_data['checklist_category']}', using first applicable")
                finding_data["checklist_category"] = task.checklist_sections[0] if task.checklist_sections else "Code Quality"
            
            # Validate line number
            try:
                line_num = int(finding_data["line"])
                if line_num < 0:
                    line_num = 1
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid line number '{finding_data['line']}', defaulting to 1")
                line_num = 1
            
            # Validate confidence score
            confidence = finding_data.get("confidence", 0.8)
            try:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    confidence = 0.8
            except (ValueError, TypeError):
                confidence = 0.8
            
            # Create Finding object
            finding = Finding(
                title=str(finding_data["title"]).strip(),
                description=str(finding_data["description"]).strip(),
                severity=finding_data["severity"],
                checklist_category=finding_data["checklist_category"],
                file=str(finding_data["file"]).strip(),
                line=line_num,
                confidence=confidence
            )
            
            return finding
            
        except Exception as e:
            self.logger.warning(f"Failed to validate finding: {e}")
            return None
    
    def _get_file_content(self, file_path: str, changed_files: List[ChangedFile]) -> str:
        """
        Get file content from the changed files list.
        
        Args:
            file_path: Path to the file
            changed_files: List of changed files with content
            
        Returns:
            str: File content or empty string if not found
        """
        for cf in changed_files:
            if cf.path == file_path:
                return cf.content
        
        self.logger.warning(f"File content not found for: {file_path}")
        return ""
    
    def review_code(self, tasks: List[ReviewTask], changed_files: List[ChangedFile]) -> ReviewerOutput:
        """
        Review code files according to the provided tasks and checklist constraints.
        
        This method applies the enterprise code review checklist as hard constraints
        and generates structured findings for each identified issue.
        
        Args:
            tasks: List of review tasks from the Planner Agent
            changed_files: List of changed files with content
            
        Returns:
            ReviewerOutput: Structured findings from code review
            
        Raises:
            BedrockError: If code review analysis fails
        """
        try:
            self.logger.info(f"Starting code review for {len(tasks)} tasks...")
            
            if not tasks:
                self.logger.info("No review tasks provided")
                return ReviewerOutput(findings=[])
            
            all_findings = []
            
            for task in tasks:
                try:
                    self.logger.info(f"Reviewing {task.file} with {task.depth} depth...")
                    
                    # Get file content
                    file_content = self._get_file_content(task.file, changed_files)
                    
                    if not file_content:
                        self.logger.warning(f"Skipping {task.file} - no content available")
                        continue
                    
                    # Skip very large files to avoid token limits
                    if len(file_content) > 50000:  # ~12k tokens
                        self.logger.warning(f"Skipping {task.file} - file too large ({len(file_content)} chars)")
                        continue
                    
                    # Create checklist prompt
                    prompt = self._create_checklist_prompt(task, file_content)
                    
                    # Get Claude's analysis
                    try:
                        claude_response = self.bedrock_client.invoke_claude(prompt, max_tokens=3000)
                        
                        # Validate response structure
                        if "findings" not in claude_response:
                            raise BedrockError(f"Invalid Claude response for {task.file}: missing 'findings' field")
                        
                        findings_data = claude_response["findings"]
                        if not isinstance(findings_data, list):
                            raise BedrockError(f"Invalid Claude response for {task.file}: 'findings' must be a list")
                        
                        # Process each finding
                        task_findings = []
                        for finding_data in findings_data:
                            finding = self._validate_finding(finding_data, task)
                            if finding:
                                task_findings.append(finding)
                        
                        all_findings.extend(task_findings)
                        
                        self.logger.info(f"Found {len(task_findings)} issues in {task.file}")
                        
                    except BedrockError as e:
                        self.logger.error(f"Claude analysis failed for {task.file}: {e}")
                        # Continue with other files rather than failing completely
                        continue
                    
                except Exception as e:
                    self.logger.error(f"Failed to review {task.file}: {e}")
                    continue
            
            # Log summary without exposing file content
            self.logger.info(f"Code review completed: {len(all_findings)} total findings")
            
            # Log findings summary by severity
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            category_counts = {}
            
            for finding in all_findings:
                severity_counts[finding.severity] += 1
                category_counts[finding.checklist_category] = category_counts.get(finding.checklist_category, 0) + 1
            
            self.logger.info(f"Findings by severity: {severity_counts}")
            self.logger.info(f"Findings by category: {category_counts}")
            
            return ReviewerOutput(findings=all_findings)
            
        except Exception as e:
            self.logger.error(f"Code review failed: {str(e)}")
            raise BedrockError(f"Reviewer Agent failed: {str(e)}")


def create_reviewer_agent(bedrock_client: BedrockClient) -> ReviewerAgent:
    """
    Create and initialize Reviewer Agent.
    
    Args:
        bedrock_client: AWS Bedrock client for Claude interactions
        
    Returns:
        ReviewerAgent: Initialized Reviewer Agent
    """
    return ReviewerAgent(bedrock_client)


# Verifier Agent Implementation
class VerifierAgent:
    """
    Verifier Agent responsible for finding validation and quality control.
    
    The Verifier Agent processes findings from the Reviewer Agent and:
    - Assigns confidence scores to findings
    - Filters out false positives and low-quality findings
    - Validates severity levels and adjusts if necessary
    - Enforces signal quality before issue creation
    - Generates structured JSON output with filtered findings
    """
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize Verifier Agent with Bedrock client.
        
        Args:
            bedrock_client: AWS Bedrock client for Claude interactions
        """
        self.bedrock_client = bedrock_client
        self.logger = logging.getLogger(__name__)
        
        # Define confidence thresholds for different severity levels
        self.confidence_thresholds = {
            "high": 0.7,    # High severity findings need high confidence
            "medium": 0.6,  # Medium severity findings need moderate confidence
            "low": 0.5      # Low severity findings need basic confidence
        }
        
        # Define patterns that indicate potential false positives
        self.false_positive_patterns = [
            "might", "could", "possibly", "potentially", "maybe",
            "consider", "suggestion", "recommendation", "optional",
            "style preference", "personal preference", "subjective"
        ]
        
        # Define patterns that indicate high-confidence findings
        self.high_confidence_patterns = [
            "vulnerability", "security risk", "injection", "exposure",
            "memory leak", "null pointer", "buffer overflow", "race condition",
            "deadlock", "infinite loop", "stack overflow", "heap corruption"
        ]
    
    def _calculate_confidence_score(self, finding: Finding) -> float:
        """
        Calculate confidence score for a finding based on various factors.
        
        Args:
            finding: Finding to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        base_confidence = finding.confidence
        
        # Adjust confidence based on title and description content
        title_lower = finding.title.lower()
        desc_lower = finding.description.lower()
        combined_text = f"{title_lower} {desc_lower}"
        
        # Reduce confidence for potential false positive patterns
        false_positive_count = sum(1 for pattern in self.false_positive_patterns 
                                 if pattern in combined_text)
        if false_positive_count > 0:
            base_confidence -= (false_positive_count * 0.1)
        
        # Increase confidence for high-confidence patterns
        high_confidence_count = sum(1 for pattern in self.high_confidence_patterns 
                                  if pattern in combined_text)
        if high_confidence_count > 0:
            base_confidence += (high_confidence_count * 0.1)
        
        # Adjust confidence based on severity-specific factors
        if finding.severity == "high":
            # High severity findings should have specific, actionable descriptions
            if len(finding.description) < 50:
                base_confidence -= 0.2  # Too brief for high severity
            if "line" in desc_lower and str(finding.line) in finding.description:
                base_confidence += 0.1  # Specific line reference increases confidence
        
        elif finding.severity == "medium":
            # Medium severity findings should have clear reasoning
            if "because" in desc_lower or "due to" in desc_lower:
                base_confidence += 0.05  # Clear reasoning increases confidence
        
        elif finding.severity == "low":
            # Low severity findings are often style/preference issues
            if any(word in combined_text for word in ["style", "format", "naming"]):
                base_confidence -= 0.1  # Style issues are less critical
        
        # Adjust confidence based on checklist category
        if finding.checklist_category == "Security":
            base_confidence += 0.1  # Security findings are high priority
        elif finding.checklist_category == "Documentation":
            base_confidence -= 0.05  # Documentation issues are lower priority
        
        # Ensure confidence stays within valid range
        return max(0.0, min(1.0, base_confidence))
    
    def _validate_severity(self, finding: Finding) -> str:
        """
        Validate and potentially adjust the severity level of a finding.
        
        Args:
            finding: Finding to validate
            
        Returns:
            str: Validated severity level
        """
        original_severity = finding.severity
        title_lower = finding.title.lower()
        desc_lower = finding.description.lower()
        combined_text = f"{title_lower} {desc_lower}"
        
        # Check for security-related keywords that should be high severity
        security_high_keywords = [
            "injection", "vulnerability", "exploit", "attack", "breach",
            "unauthorized", "privilege escalation", "authentication bypass"
        ]
        
        if any(keyword in combined_text for keyword in security_high_keywords):
            if original_severity != "high":
                self.logger.info(f"Upgrading severity to 'high' for security finding: {finding.title}")
                return "high"
        
        # Check for performance issues that should be medium severity
        performance_keywords = [
            "memory leak", "performance", "slow", "timeout", "bottleneck",
            "inefficient", "optimization", "scalability"
        ]
        
        if any(keyword in combined_text for keyword in performance_keywords):
            if original_severity == "low":
                self.logger.info(f"Upgrading severity to 'medium' for performance finding: {finding.title}")
                return "medium"
        
        # Check for style/documentation issues that should be low severity
        style_keywords = [
            "style", "formatting", "naming convention", "comment", "documentation",
            "whitespace", "indentation", "spelling"
        ]
        
        if any(keyword in combined_text for keyword in style_keywords):
            if original_severity == "high":
                self.logger.info(f"Downgrading severity to 'medium' for style finding: {finding.title}")
                return "medium"
            elif original_severity == "medium" and finding.checklist_category == "Documentation":
                self.logger.info(f"Downgrading severity to 'low' for documentation finding: {finding.title}")
                return "low"
        
        return original_severity
    
    def _is_false_positive(self, finding: Finding) -> bool:
        """
        Determine if a finding is likely a false positive.
        
        Args:
            finding: Finding to evaluate
            
        Returns:
            bool: True if likely false positive, False otherwise
        """
        # Check confidence against severity-specific thresholds
        required_confidence = self.confidence_thresholds.get(finding.severity, 0.5)
        if finding.confidence < required_confidence:
            self.logger.info(f"Filtering out low-confidence finding: {finding.title} (confidence: {finding.confidence:.2f})")
            return True
        
        # Check for vague or non-specific descriptions
        if len(finding.description) < 20:
            self.logger.info(f"Filtering out vague finding: {finding.title}")
            return True
        
        # Check for excessive false positive patterns
        combined_text = f"{finding.title.lower()} {finding.description.lower()}"
        false_positive_count = sum(1 for pattern in self.false_positive_patterns 
                                 if pattern in combined_text)
        
        if false_positive_count >= 3:
            self.logger.info(f"Filtering out finding with too many uncertain patterns: {finding.title}")
            return True
        
        # Check for duplicate or very similar findings (basic check)
        # This would be enhanced with more sophisticated similarity detection
        if finding.title.lower().count("todo") > 0 and finding.severity == "high":
            self.logger.info(f"Filtering out TODO comment marked as high severity: {finding.title}")
            return True
        
        return False
    
    def _create_verification_prompt(self, findings: List[Finding]) -> str:
        """
        Create prompt for Claude to verify and refine findings.
        
        Args:
            findings: List of findings to verify
            
        Returns:
            str: Formatted prompt for Claude
        """
        # Create findings summary without exposing source code
        findings_summary = []
        for i, finding in enumerate(findings):
            summary = {
                "id": i,
                "title": finding.title,
                "description": finding.description,
                "severity": finding.severity,
                "checklist_category": finding.checklist_category,
                "file": finding.file,
                "line": finding.line,
                "confidence": finding.confidence
            }
            findings_summary.append(summary)
        
        prompt = f"""You are a senior code review expert performing quality control on code review findings.

FINDINGS TO VERIFY:
{json.dumps(findings_summary, indent=2)}

Your task is to verify each finding and filter out false positives while preserving genuine issues.

VERIFICATION CRITERIA:
1. Confidence Assessment:
   - High severity findings need confidence ≥ 0.7
   - Medium severity findings need confidence ≥ 0.6
   - Low severity findings need confidence ≥ 0.5

2. False Positive Indicators:
   - Vague or non-specific descriptions
   - Overly cautious language ("might", "could", "possibly")
   - Style preferences rather than actual issues
   - Duplicate or redundant findings

3. Genuine Issue Indicators:
   - Specific security vulnerabilities
   - Clear performance problems
   - Definite bugs or errors
   - Architectural violations
   - Missing error handling

4. Severity Validation:
   - High: Security vulnerabilities, critical bugs, major architectural flaws
   - Medium: Performance issues, maintainability concerns, minor bugs
   - Low: Style issues, minor improvements, documentation gaps

For each finding, determine:
- Whether it should be kept (not a false positive)
- Adjusted confidence score (0.0-1.0)
- Validated severity level
- Brief reasoning for decision

Return ONLY a valid JSON object with this exact structure:
{{
  "final_issues": [
    {{
      "id": 0,
      "keep": true,
      "confidence": 0.85,
      "severity": "high",
      "reasoning": "Specific security vulnerability with clear impact"
    }}
  ]
}}

Only include findings that should be kept. Exclude false positives entirely.
Do not include any explanation outside the JSON object."""
        
        return prompt
    
    def verify_findings(self, findings: List[Finding]) -> VerifierOutput:
        """
        Verify findings and filter out false positives while preserving genuine issues.
        
        This method applies quality control to findings from the Reviewer Agent:
        - Calculates confidence scores based on content analysis
        - Validates and adjusts severity levels
        - Filters out likely false positives
        - Uses Claude for additional verification when needed
        
        Args:
            findings: List of findings from Reviewer Agent
            
        Returns:
            VerifierOutput: Filtered and verified findings
            
        Raises:
            BedrockError: If verification process fails
        """
        try:
            self.logger.info(f"Starting verification of {len(findings)} findings...")
            
            if not findings:
                self.logger.info("No findings to verify")
                return VerifierOutput(final_issues=[])
            
            # Step 1: Calculate confidence scores and validate severity
            processed_findings = []
            for finding in findings:
                # Calculate new confidence score
                new_confidence = self._calculate_confidence_score(finding)
                
                # Validate severity
                validated_severity = self._validate_severity(finding)
                
                # Create updated finding
                updated_finding = Finding(
                    title=finding.title,
                    description=finding.description,
                    severity=validated_severity,
                    checklist_category=finding.checklist_category,
                    file=finding.file,
                    line=finding.line,
                    confidence=new_confidence
                )
                
                processed_findings.append(updated_finding)
            
            # Step 2: Filter out obvious false positives
            filtered_findings = []
            for finding in processed_findings:
                if not self._is_false_positive(finding):
                    filtered_findings.append(finding)
            
            self.logger.info(f"After initial filtering: {len(filtered_findings)} findings remain")
            
            if not filtered_findings:
                self.logger.info("All findings filtered out as false positives")
                return VerifierOutput(final_issues=[])
            
            # Step 3: Use Claude for additional verification on remaining findings
            final_findings = []
            
            try:
                # Only use Claude verification if we have a reasonable number of findings
                if len(filtered_findings) <= 20:  # Avoid token limit issues
                    prompt = self._create_verification_prompt(filtered_findings)
                    claude_response = self.bedrock_client.invoke_claude(prompt, max_tokens=2000)
                    
                    # Validate Claude's response
                    if "final_issues" not in claude_response:
                        raise BedrockError("Invalid Claude response: missing 'final_issues' field")
                    
                    claude_decisions = claude_response["final_issues"]
                    if not isinstance(claude_decisions, list):
                        raise BedrockError("Invalid Claude response: 'final_issues' must be a list")
                    
                    # Apply Claude's decisions
                    for decision in claude_decisions:
                        try:
                            finding_id = decision.get("id")
                            if finding_id is None or finding_id >= len(filtered_findings):
                                continue
                            
                            if decision.get("keep", False):
                                original_finding = filtered_findings[finding_id]
                                
                                # Apply Claude's adjustments
                                adjusted_confidence = decision.get("confidence", original_finding.confidence)
                                adjusted_severity = decision.get("severity", original_finding.severity)
                                
                                # Validate adjustments
                                if not 0.0 <= adjusted_confidence <= 1.0:
                                    adjusted_confidence = original_finding.confidence
                                
                                if adjusted_severity not in ["high", "medium", "low"]:
                                    adjusted_severity = original_finding.severity
                                
                                # Create final finding
                                final_finding = Finding(
                                    title=original_finding.title,
                                    description=original_finding.description,
                                    severity=adjusted_severity,
                                    checklist_category=original_finding.checklist_category,
                                    file=original_finding.file,
                                    line=original_finding.line,
                                    confidence=adjusted_confidence
                                )
                                
                                final_findings.append(final_finding)
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to process Claude decision: {e}")
                            continue
                
                else:
                    # Too many findings for Claude verification, use rule-based filtering only
                    self.logger.info(f"Too many findings ({len(filtered_findings)}) for Claude verification, using rule-based filtering")
                    final_findings = filtered_findings
                
            except BedrockError as e:
                self.logger.warning(f"Claude verification failed, using rule-based filtering: {e}")
                final_findings = filtered_findings
            
            # Step 4: Final quality check and logging
            if not final_findings:
                final_findings = filtered_findings  # Fallback to rule-based filtering
            
            # Log verification summary
            self.logger.info(f"Verification completed: {len(final_findings)} final issues")
            
            # Log summary by severity and category
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            category_counts = {}
            confidence_sum = 0.0
            
            for finding in final_findings:
                severity_counts[finding.severity] += 1
                category_counts[finding.checklist_category] = category_counts.get(finding.checklist_category, 0) + 1
                confidence_sum += finding.confidence
            
            avg_confidence = confidence_sum / len(final_findings) if final_findings else 0.0
            
            self.logger.info(f"Final issues by severity: {severity_counts}")
            self.logger.info(f"Final issues by category: {category_counts}")
            self.logger.info(f"Average confidence score: {avg_confidence:.2f}")
            
            return VerifierOutput(final_issues=final_findings)
            
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            raise BedrockError(f"Verifier Agent failed: {str(e)}")


def create_verifier_agent(bedrock_client: BedrockClient) -> VerifierAgent:
    """
    Create and initialize Verifier Agent.
    
    Args:
        bedrock_client: AWS Bedrock client for Claude interactions
        
    Returns:
        VerifierAgent: Initialized Verifier Agent
    """
    return VerifierAgent(bedrock_client)

# GitHub Integration Module
class GitHubError(Exception):
    """Custom exception for GitHub-related errors"""
    pass


def post_review_comment(config: Dict[str, str], findings: List[Finding]) -> bool:
    """
    Post code review findings as GitHub PR comments.
    
    Args:
        config: Environment configuration with GitHub credentials
        findings: List of verified findings to post
        
    Returns:
        bool: True if comments posted successfully, False otherwise
        
    Raises:
        GitHubError: If GitHub API operations fail
    """
    try:
        if not findings:
            logger.info("No findings to post as GitHub comments")
            return True
        
        # Get GitHub context from environment
        github_token = config['GITHUB_TOKEN']
        github_repo = config['GITHUB_REPOSITORY']
        github_actor = config['GITHUB_ACTOR']
        
        # Get PR number from GitHub event (if running in GitHub Actions)
        pr_number = os.environ.get('GITHUB_PR_NUMBER')
        github_sha = os.environ.get('GITHUB_SHA', 'HEAD')
        
        if not pr_number:
            # Try to extract from GITHUB_REF if it's a PR
            github_ref = os.environ.get('GITHUB_REF', '')
            if github_ref.startswith('refs/pull/'):
                pr_number = github_ref.split('/')[2]
        
        if not pr_number:
            logger.warning("No PR number found - posting as commit comments instead")
            return post_commit_comments(config, findings, github_sha)
        
        # GitHub API headers
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': f'AI-Code-Review-Agent/{VERSION}'
        }
        
        # Group findings by file for better organization
        findings_by_file = {}
        for finding in findings:
            if finding.file not in findings_by_file:
                findings_by_file[finding.file] = []
            findings_by_file[finding.file].append(finding)
        
        # Post review comments
        api_base = f'https://api.github.com/repos/{github_repo}'
        comments_posted = 0
        
        for file_path, file_findings in findings_by_file.items():
            for finding in file_findings:
                # Create review comment body
                severity_emoji = {
                    'high': '🚨',
                    'medium': '⚠️', 
                    'low': '💡'
                }
                
                comment_body = f"""{severity_emoji.get(finding.severity, '📝')} **{finding.title}**

**Category:** {finding.checklist_category}
**Severity:** {finding.severity.upper()}
**Confidence:** {finding.confidence:.0%}

{finding.description}

---
*Generated by AI Code Review Agent v{VERSION}*"""
                
                # Prepare comment data
                comment_data = {
                    'body': comment_body,
                    'path': finding.file,
                    'line': finding.line,
                    'side': 'RIGHT'
                }
                
                # Post the comment
                try:
                    response = requests.post(
                        f'{api_base}/pulls/{pr_number}/comments',
                        headers=headers,
                        json=comment_data,
                        timeout=30
                    )
                    
                    if response.status_code == 201:
                        comments_posted += 1
                        logger.info(f"Posted comment for {finding.title} in {finding.file}:{finding.line}")
                    else:
                        logger.warning(f"Failed to post comment: {response.status_code} - {response.text}")
                
                except requests.RequestException as e:
                    logger.error(f"Request failed for comment on {finding.file}:{finding.line}: {e}")
                    continue
        
        logger.info(f"Successfully posted {comments_posted} review comments to PR #{pr_number}")
        return comments_posted > 0
        
    except Exception as e:
        raise GitHubError(f"Failed to post GitHub review comments: {str(e)}")


def post_commit_comments(config: Dict[str, str], findings: List[Finding], sha: str) -> bool:
    """
    Post findings as commit comments when not in a PR context.
    
    Args:
        config: Environment configuration
        findings: List of findings to post
        sha: Git commit SHA
        
    Returns:
        bool: True if comments posted successfully
    """
    try:
        github_token = config['GITHUB_TOKEN']
        github_repo = config['GITHUB_REPOSITORY']
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': f'AI-Code-Review-Agent/{VERSION}'
        }
        
        api_base = f'https://api.github.com/repos/{github_repo}'
        comments_posted = 0
        
        for finding in findings:
            severity_emoji = {
                'high': '🚨',
                'medium': '⚠️',
                'low': '💡'
            }
            
            comment_body = f"""{severity_emoji.get(finding.severity, '📝')} **{finding.title}**

**File:** {finding.file}:{finding.line}
**Category:** {finding.checklist_category}
**Severity:** {finding.severity.upper()}

{finding.description}

---
*Generated by AI Code Review Agent v{VERSION}*"""
            
            comment_data = {
                'body': comment_body,
                'path': finding.file,
                'line': finding.line
            }
            
            try:
                response = requests.post(
                    f'{api_base}/commits/{sha}/comments',
                    headers=headers,
                    json=comment_data,
                    timeout=30
                )
                
                if response.status_code == 201:
                    comments_posted += 1
                    logger.info(f"Posted commit comment for {finding.title}")
                else:
                    logger.warning(f"Failed to post commit comment: {response.status_code}")
            
            except requests.RequestException as e:
                logger.error(f"Request failed for commit comment: {e}")
                continue
        
        logger.info(f"Successfully posted {comments_posted} commit comments")
        return comments_posted > 0
        
    except Exception as e:
        logger.error(f"Failed to post commit comments: {e}")
        return False


def create_review_summary(findings: List[Finding]) -> str:
    """
    Create a summary of the code review findings.
    
    Args:
        findings: List of verified findings
        
    Returns:
        str: Formatted summary text
    """
    if not findings:
        return "✅ No issues found in the code review."
    
    # Count findings by severity
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    category_counts = {}
    
    for finding in findings:
        severity_counts[finding.severity] += 1
        category_counts[finding.checklist_category] = category_counts.get(finding.checklist_category, 0) + 1
    
    # Create summary
    total_issues = len(findings)
    summary_lines = [
        f"## 🤖 AI Code Review Summary",
        f"",
        f"**Total Issues Found:** {total_issues}",
        f"",
        f"### Issues by Severity",
        f"- 🚨 **High:** {severity_counts['high']} issues",
        f"- ⚠️ **Medium:** {severity_counts['medium']} issues", 
        f"- 💡 **Low:** {severity_counts['low']} issues",
        f"",
        f"### Issues by Category"
    ]
    
    for category, count in sorted(category_counts.items()):
        summary_lines.append(f"- **{category}:** {count} issues")
    
    summary_lines.extend([
        f"",
        f"### Recommendations",
        f"1. **High severity issues** should be addressed immediately",
        f"2. **Medium severity issues** should be reviewed and planned for resolution",
        f"3. **Low severity issues** can be addressed as time permits",
        f"",
        f"---",
        f"*Generated by AI Code Review Agent v{VERSION}*"
    ])
    
    return "\n".join(summary_lines)


def main_execution():
    """
    Main execution flow for the AI Code Review Agent.
    
    This function orchestrates the entire code review process:
    1. Validates environment and initializes clients
    2. Extracts changed files from git diff
    3. Plans review strategy using Planner Agent
    4. Performs detailed code analysis using Reviewer Agent
    5. Verifies and filters findings using Verifier Agent
    6. Posts results to GitHub as PR/commit comments
    """
    try:
        logger.info(f"AI Code Review Agent v{VERSION} - Starting main execution")
        
        # Step 1: Environment validation and client initialization
        logger.info("Step 1: Validating environment and initializing clients...")
        config = validate_startup_environment()
        
        # Initialize AWS Bedrock client
        bedrock_client = create_bedrock_client(config)
        logger.info("AWS Bedrock client initialized successfully")
        
        # Step 2: Extract changed files from git diff
        logger.info("Step 2: Extracting changed files from git diff...")
        changed_files = get_changed_files()
        
        if not changed_files:
            logger.info("No changed files found - nothing to review")
            print("✅ No changes detected - code review complete")
            return
        
        logger.info(f"Found {len(changed_files)} changed files to review")
        
        # Step 3: Plan review strategy using Planner Agent
        logger.info("Step 3: Planning review strategy...")
        planner = create_planner_agent(bedrock_client)
        planner_output = planner.plan_review(changed_files)
        
        if not planner_output.tasks:
            logger.info("No review tasks generated - nothing to review")
            print("✅ No review tasks needed - code review complete")
            return
        
        logger.info(f"Generated {len(planner_output.tasks)} review tasks")
        
        # Step 4: Perform detailed code analysis using Reviewer Agent
        logger.info("Step 4: Performing detailed code analysis...")
        reviewer = create_reviewer_agent(bedrock_client)
        reviewer_output = reviewer.review_code(planner_output.tasks, changed_files)
        
        if not reviewer_output.findings:
            logger.info("No findings from code review")
            print("✅ No issues found - code review complete")
            return
        
        logger.info(f"Found {len(reviewer_output.findings)} potential issues")
        
        # Step 5: Verify and filter findings using Verifier Agent
        logger.info("Step 5: Verifying and filtering findings...")
        verifier = create_verifier_agent(bedrock_client)
        verifier_output = verifier.verify_findings(reviewer_output.findings)
        
        final_issues = verifier_output.final_issues
        logger.info(f"Verified {len(final_issues)} final issues")
        
        # Step 6: Create summary and post results to GitHub
        logger.info("Step 6: Posting results to GitHub...")
        
        # Create and log summary
        summary = create_review_summary(final_issues)
        print("\n" + summary)
        
        # Post comments to GitHub
        if final_issues:
            try:
                success = post_review_comment(config, final_issues)
                if success:
                    logger.info("Successfully posted review comments to GitHub")
                    print(f"\n🎉 Posted {len(final_issues)} review comments to GitHub")
                else:
                    logger.warning("Failed to post some or all review comments")
                    print("\n⚠️ Some review comments may not have been posted")
            
            except GitHubError as e:
                logger.error(f"GitHub integration failed: {e}")
                print(f"\n❌ Failed to post comments to GitHub: {e}")
                # Don't fail the entire process if GitHub posting fails
        
        else:
            logger.info("No final issues to post to GitHub")
            print("\n✅ No issues found - code review complete")
        
        # Log final statistics
        logger.info("Code review process completed successfully")
        logger.info(f"Final statistics: {len(changed_files)} files, {len(planner_output.tasks)} tasks, {len(final_issues)} issues")
        
        print(f"\n📊 Review Statistics:")
        print(f"   Files analyzed: {len(changed_files)}")
        print(f"   Review tasks: {len(planner_output.tasks)}")
        print(f"   Issues found: {len(reviewer_output.findings)}")
        print(f"   Final issues: {len(final_issues)}")
        
    except EnvironmentError as e:
        logger.error(f"Environment configuration error: {e}")
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
        
    except GitError as e:
        logger.error(f"Git operation error: {e}")
        print(f"❌ Git Error: {e}")
        sys.exit(1)
        
    except BedrockError as e:
        logger.error(f"AWS Bedrock error: {e}")
        print(f"❌ AWS Bedrock Error: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    logger.info(f"AI Code Review Agent v{VERSION} starting...")
    
    # Validate environment configuration at startup
    try:
        config = validate_startup_environment()
        logger.info("Environment configuration validated successfully")
        
        # Execute main code review process
        main_execution()
        
    except SystemExit:
        # Environment validation failed - exit gracefully
        raise
    except Exception as e:
        logger.error(f"Unexpected startup error: {str(e)}")
        sys.exit(1)