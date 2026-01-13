#!/usr/bin/env python3
"""
File handler with path traversal and error handling issues
"""

def read_user_file(filename):
    # Path traversal vulnerability - no validation
    file_path = "/uploads/" + filename
    
    # Missing exception handling - could crash
    with open(file_path, 'r') as f:
        content = f.read()
    
    return content

def process_upload(file_data, filename):
    # No file type validation
    # No size limits
    # No malware scanning
    
    save_path = "/var/uploads/" + filename  # Another path traversal issue
    
    # Writing without proper error handling
    with open(save_path, 'wb') as f:
        f.write(file_data)
    
    return save_path

def delete_file(filename):
    import os
    
    # Command injection vulnerability
    os.system(f"rm /uploads/{filename}")  # User input in system command!
    
    return True