#!/usr/bin/env python3
"""
Example code with various issues for AI Code Review Agent to detect
"""

import os
import sqlite3

def login_user(username, password):
    # Security issue: SQL injection vulnerability
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    result = cursor.execute(query).fetchone()
    
    # Error handling issue: No exception handling
    # Performance issue: Connection not closed properly
    
    if result:
        return True
    return False

def process_file(filename):
    # Security issue: Path traversal vulnerability
    file_path = "/uploads/" + filename
    
    # Error handling issue: No file existence check
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Code quality issue: Magic number
    if len(content) > 1000000:
        return "File too large"
    
    return content

class UserManager:
    def __init__(self):
        # Architecture issue: Hardcoded credentials
        self.admin_password = "admin123"
        self.users = []
    
    def add_user(self, user_data):
        # Error handling issue: No input validation
        self.users.append(user_data)
    
    def get_user_by_id(self, user_id):
        # Performance issue: Linear search instead of hash lookup
        for user in self.users:
            if user['id'] == user_id:
                return user
        return None

# Documentation issue: Missing docstrings and comments
def calculate_hash(data):
    import hashlib
    return hashlib.md5(data.encode()).hexdigest()  # Security issue: MD5 is weak

# Testing issue: No unit tests for critical functions
if __name__ == "__main__":
    # Maintainability issue: No configuration management
    print("Starting application...")
    manager = UserManager()
    print("Application started")