#!/bin/bash
set -e

# Install the package and dependencies
pip install -e .[all]
pip install jupyter

# Install Claude CLI
curl -fsSL https://claude.ai/install.sh | bash
