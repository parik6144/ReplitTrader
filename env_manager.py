
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv, set_key

class EnvManager:
    def __init__(self):
        self.env_file = find_dotenv() or '.env'
        load_dotenv(self.env_file)
        
        # Create .env file if it doesn't exist
        if not os.path.exists(self.env_file):
            Path(self.env_file).touch()
    
    def set_env(self, key, value):
        """Set environment variable in .env file"""
        try:
            set_key(self.env_file, key, str(value))
            os.environ[key] = str(value)
            return True
        except Exception as e:
            print(f"Error setting {key}: {e}")
            return False
    
    def get_env(self, key, default=None):
        """Get environment variable"""
        return os.getenv(key, default)
    
    def delete_env(self, key):
        """Delete environment variable from .env file"""
        try:
            # Read the file
            if os.path.exists(self.env_file):
                with open(self.env_file, 'r') as file:
                    lines = file.readlines()
                
                # Filter out the key
                with open(self.env_file, 'w') as file:
                    for line in lines:
                        if not line.strip().startswith(f"{key}="):
                            file.write(line)
                
                # Remove from current environment
                if key in os.environ:
                    del os.environ[key]
                
                return True
        except Exception as e:
            print(f"Error deleting {key}: {e}")
            return False
    
    def list_all_keys(self):
        """List all keys in .env file"""
        keys = []
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as file:
                for line in file:
                    if '=' in line and not line.strip().startswith('#'):
                        key = line.split('=')[0].strip()
                        keys.append(key)
        return keys
    
    def backup_env(self):
        """Create backup of .env file"""
        try:
            import shutil
            backup_file = f"{self.env_file}.backup"
            shutil.copy2(self.env_file, backup_file)
            return backup_file
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
