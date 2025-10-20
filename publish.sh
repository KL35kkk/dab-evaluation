#!/bin/bash
"""
Publish dab-evaluation to PyPI
"""

echo "🚀 Publishing dab-evaluation to PyPI"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found. Please run this script from the project root."
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf dab_eval.egg-info/

# Install build tools
echo "📦 Installing build tools..."
pip install --upgrade pip
pip install build twine

# Build the package
echo "🔨 Building package..."
python -m build

# Check the build
echo "🔍 Checking build..."
twine check dist/*

# Upload to PyPI (test first)
echo "📤 Uploading to PyPI Test..."
echo "Please enter your PyPI credentials:"
twine upload --repository testpypi dist/*

echo "✅ Package uploaded to PyPI Test!"
echo "You can test install with: pip install -i https://test.pypi.org/simple/ dab-evaluation"

# Ask if user wants to upload to production PyPI
read -p "Do you want to upload to production PyPI? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to production PyPI..."
    twine upload dist/*
    echo "✅ Package uploaded to production PyPI!"
    echo "You can install with: pip install dab-evaluation"
else
    echo "⏭️  Skipping production upload"
fi

echo "🎉 Publishing complete!"
