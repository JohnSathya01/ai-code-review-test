#!/usr/bin/env python3
"""
Test file with security vulnerabilities for AI Code Review testing
"""

def login_user(username, password):
    # SQL Injection vulnerability - user input directly in query
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    
    # Missing error handling
    result = execute_sql(query)
    
    if result:
        return True
    return False

def execute_sql(query):
    # Simulate database execution
    import sqlite3
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    return cursor.execute(query).fetchone()

# Hardcoded API key - security issue
API_KEY = "sk-1234567890abcdef"

def authenticate_api(user_token):
    # Weak comparison - timing attack vulnerability
    if user_token == API_KEY:
        return True
    return False