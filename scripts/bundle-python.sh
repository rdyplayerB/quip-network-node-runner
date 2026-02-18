#!/bin/bash
#
# Bundle Python 3.13 standalone interpreter with dependencies for Tauri app
#
set -e

# Configuration
PYTHON_VERSION="3.13.2"
PYTHON_SHORT="3.13"
PBS_DATE="20250212"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/python-bundle"
RESOURCES_DIR="$PROJECT_ROOT/src-tauri/resources"

echo "=== Quip Network Node - Python Bundle Script ==="
echo "Project root: $PROJECT_ROOT"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    PLATFORM="aarch64-apple-darwin"
    echo "Architecture: Apple Silicon (ARM64)"
elif [ "$ARCH" = "x86_64" ]; then
    PLATFORM="x86_64-apple-darwin"
    echo "Architecture: Intel (x86_64)"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

# Clean and create directories
echo ""
echo "=== Step 1: Preparing directories ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$RESOURCES_DIR"

# Download python-build-standalone
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_DATE/cpython-$PYTHON_VERSION+$PBS_DATE-$PLATFORM-install_only.tar.gz"
echo ""
echo "=== Step 2: Downloading Python $PYTHON_VERSION ==="
echo "URL: $PBS_URL"

if [ -f "$BUILD_DIR/python.tar.gz" ]; then
    echo "Using cached download..."
else
    curl -L -o "$BUILD_DIR/python.tar.gz" "$PBS_URL"
fi

# Extract Python
echo ""
echo "=== Step 3: Extracting Python ==="
tar -xzf "$BUILD_DIR/python.tar.gz" -C "$BUILD_DIR"

# Strip unnecessary files from Python to reduce size
echo ""
echo "=== Step 4: Stripping Python distribution ==="
cd "$BUILD_DIR/python"

# Remove components not needed at runtime
rm -rf include/                                          # Headers (for compiling extensions)
rm -rf lib/python$PYTHON_SHORT/test/                     # Test suite
rm -rf lib/python$PYTHON_SHORT/*/tests/                  # Package tests
rm -rf lib/python$PYTHON_SHORT/*/test/                   # Package tests
rm -rf lib/python$PYTHON_SHORT/tkinter/                  # Tkinter GUI
rm -rf lib/python$PYTHON_SHORT/lib-dynload/_tkinter*     # Tkinter bindings
rm -rf lib/python$PYTHON_SHORT/idlelib/                  # IDLE
rm -rf lib/python$PYTHON_SHORT/turtle*                   # Turtle graphics
rm -rf lib/python$PYTHON_SHORT/turtledemo/               # Turtle demos
rm -rf lib/python$PYTHON_SHORT/ensurepip/                # We'll install packages ourselves
rm -rf share/                                            # Documentation

# Remove __pycache__ (will be regenerated)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Install pip using get-pip.py (works without ensurepip)
echo ""
echo "=== Step 5: Installing pip and dependencies ==="
curl -sS https://bootstrap.pypa.io/get-pip.py -o "$BUILD_DIR/get-pip.py"
"$BUILD_DIR/python/bin/python3" "$BUILD_DIR/get-pip.py" --no-warn-script-location

echo "Installing dependencies from requirements.txt..."
"$BUILD_DIR/python/bin/python3" -m pip install -r "$PROJECT_ROOT/node-extract/requirements.txt" --no-warn-script-location

# Prepare site-packages
echo ""
echo "=== Step 6: Preparing site-packages ==="
SITE_PACKAGES="$BUILD_DIR/python/lib/python$PYTHON_SHORT/site-packages"

# Remove packages not needed at runtime
rm -rf "$SITE_PACKAGES/pip"*
rm -rf "$SITE_PACKAGES/setuptools"*
rm -rf "$SITE_PACKAGES/_distutils_hack"
rm -rf "$SITE_PACKAGES/wheel"*
rm -rf "$SITE_PACKAGES/pkg_resources"

# Remove test directories
find "$SITE_PACKAGES" -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove __pycache__
find "$SITE_PACKAGES" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -name "*.pyc" -delete 2>/dev/null || true

# Copy to resources directory
echo ""
echo "=== Step 7: Copying to resources directory ==="

# Remove old resources
rm -rf "$RESOURCES_DIR/python"
rm -rf "$RESOURCES_DIR/site-packages"
rm -rf "$RESOURCES_DIR/quip-node"

# Copy Python interpreter
echo "Copying Python interpreter..."
cp -R "$BUILD_DIR/python" "$RESOURCES_DIR/python"

# Copy site-packages
echo "Copying site-packages..."
cp -R "$SITE_PACKAGES" "$RESOURCES_DIR/site-packages"

# Copy application code
echo "Copying application code..."
mkdir -p "$RESOURCES_DIR/quip-node"
cp "$PROJECT_ROOT/node-extract/quip_cli.py" "$RESOURCES_DIR/quip-node/"
cp -R "$PROJECT_ROOT/node-extract/shared" "$RESOURCES_DIR/quip-node/"
cp -R "$PROJECT_ROOT/node-extract/CPU" "$RESOURCES_DIR/quip-node/"
cp -R "$PROJECT_ROOT/node-extract/GPU" "$RESOURCES_DIR/quip-node/"
cp -R "$PROJECT_ROOT/node-extract/QPU" "$RESOURCES_DIR/quip-node/"
cp -R "$PROJECT_ROOT/node-extract/dwave_topologies" "$RESOURCES_DIR/quip-node/"

# Copy genesis block
cp "$PROJECT_ROOT/node-extract/genesis_block_public.json" "$RESOURCES_DIR/"

# Remove __pycache__ from quip-node
find "$RESOURCES_DIR/quip-node" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Report sizes
echo ""
echo "=== Bundle Complete ==="
echo ""
echo "Bundle sizes:"
du -sh "$RESOURCES_DIR/python" 2>/dev/null || echo "  python: not found"
du -sh "$RESOURCES_DIR/site-packages" 2>/dev/null || echo "  site-packages: not found"
du -sh "$RESOURCES_DIR/quip-node" 2>/dev/null || echo "  quip-node: not found"
echo ""
echo "Total resources:"
du -sh "$RESOURCES_DIR"

echo ""
echo "=== Testing Python bundle ==="
cd "$RESOURCES_DIR"

# Test basic execution
echo "Testing Python execution..."
PYTHONHOME=./python PYTHONPATH=./site-packages:./quip-node \
  ./python/bin/python3 --version

# Test critical imports
echo "Testing imports..."
PYTHONHOME=./python PYTHONPATH=./site-packages:./quip-node \
  ./python/bin/python3 -c "
import sys
print('Python:', sys.version)
print('Testing imports...')

import click
print('  click: OK')

import numpy
print('  numpy: OK')

import aiohttp
print('  aiohttp: OK')

import cryptography
print('  cryptography: OK')

# These are the problematic D-Wave imports
import dimod
print('  dimod: OK')

from shared.version import VERSION
print(f'  shared.version: OK (v{VERSION})')

print('')
print('All critical imports successful!')
"

echo ""
echo "=== Python bundle ready! ==="
echo ""
echo "Next steps:"
echo "  1. Run 'npm run build' to build the Tauri app"
echo "  2. The app will be in src-tauri/target/release/bundle/"
