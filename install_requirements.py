#!/usr/bin/env python3
"""
Enhanced CADENCE Requirements Installer
Ensures all dependencies are properly installed for the demo
"""

import subprocess
import sys
import importlib
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_requirements():
    """Check and install requirements from requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, you have {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_core_dependencies():
    """Check if core dependencies can be imported"""
    core_deps = [
        'torch',
        'numpy', 
        'pandas',
        'sklearn',
        'fastapi',
        'uvicorn',
        'structlog',
        'sqlalchemy',
        'pydantic'
    ]
    
    print("🔍 Checking core dependencies...")
    
    for dep in core_deps:
        try:
            importlib.import_module(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - not found")
            return False
    
    return True

def check_optional_dependencies():
    """Check optional dependencies"""
    optional_deps = {
        'redis': 'Redis (caching will be disabled)',
        'datasets': 'HuggingFace datasets (will use synthetic data)',
        'transformers': 'Transformers (will use basic embeddings)'
    }
    
    print("🔍 Checking optional dependencies...")
    
    for dep, description in optional_deps.items():
        try:
            importlib.import_module(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ⚠️  {dep} - {description}")

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'static', 'logs', 'data']
    
    print("📁 Creating directories...")
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")

def main():
    """Main installation function"""
    print("🚀 Enhanced CADENCE Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not check_and_install_requirements():
        return False
    
    # Check core dependencies
    if not check_core_dependencies():
        print("\n❌ Some core dependencies failed to install properly.")
        print("Please try running: pip install -r requirements.txt")
        return False
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now run the demo with:")
    print("  python run_demo.py")
    print("\nOr manually start with:")
    print("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 