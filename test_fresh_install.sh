#!/bin/bash
# Test script for fresh installation in a virtual environment
# This simulates what a user would experience when installing from GitHub

set -e

echo "================================================"
echo "Testing Fresh Installation Process"
echo "================================================"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create a temporary virtual environment
TEMP_DIR=$(mktemp -d)
echo "Creating temporary virtual environment in $TEMP_DIR..."

python3 -m venv "$TEMP_DIR/venv"
source "$TEMP_DIR/venv/bin/activate"

echo -e "${GREEN}✓ Virtual environment created${NC}"
echo

# Test pip installation
echo "Testing: pip install git+https://github.com/marco-willi/clip-cues.git"
echo "Note: Testing with local path instead..."
echo

pip install -q /workspaces/clip-cues

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Installation successful${NC}"
else
    echo -e "${RED}✗ Installation failed${NC}"
    deactivate
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo

# Test basic import
echo "Testing: from clip_cues import load_clip_classifier"
python -c "from clip_cues import load_clip_classifier; print('✓ Import successful')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Import test passed${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    deactivate
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo

# Test quick start example
echo "Testing Quick Start example from README..."
python << 'EOF'
from clip_cues import load_clip_classifier
from PIL import Image
import numpy as np
import tempfile
import os

# Create a test image
img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
    img.save(tmp.name)
    tmp_path = tmp.name

try:
    # Load model (using relative path from installation location)
    model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")

    # Predict on the test image
    prob = model.predict(tmp_path)

    print(f"Synthetic probability: {prob:.1%}")
    print("Prediction:", "Synthetic" if prob > 0.5 else "Real")
    print("✓ Quick start example works!")

finally:
    os.unlink(tmp_path)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Quick start example passed${NC}"
else
    echo -e "${RED}✗ Quick start example failed${NC}"
    deactivate
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo

# Test batch inference
echo "Testing Batch Inference example..."
python << 'EOF'
from clip_cues import load_clip_classifier
from PIL import Image
import numpy as np
import tempfile
import os

# Create test images
tmp_files = []
for i in range(3):
    img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(tmp.name)
    tmp_files.append(tmp.name)
    tmp.close()

try:
    model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")

    # Process multiple images
    probs = model.predict_batch(tmp_files, batch_size=32)

    for path, prob in zip(tmp_files, probs):
        print(f"{os.path.basename(path)}: {prob:.1%} synthetic")

    print("✓ Batch inference example works!")

finally:
    for f in tmp_files:
        os.unlink(f)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Batch inference example passed${NC}"
else
    echo -e "${RED}✗ Batch inference example failed${NC}"
    deactivate
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo

# Test list_available_models
echo "Testing list_available_models..."
python << 'EOF'
from clip_cues import list_available_models

models = list_available_models()
print(f"Found {len(models)} available models:")
for name, info in list(models.items())[:3]:
    print(f"  - {name}: {info['description']}")
print("  ...")
print("✓ list_available_models works!")
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ list_available_models passed${NC}"
else
    echo -e "${RED}✗ list_available_models failed${NC}"
    deactivate
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Clean up
deactivate
rm -rf "$TEMP_DIR"

echo
echo "================================================"
echo -e "${GREEN}✓ All fresh installation tests passed!${NC}"
echo "================================================"
