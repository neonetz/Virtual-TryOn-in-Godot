#!/bin/bash
# Setup script for UDP Face Detection Server
# Usage: bash setup_udp_server.sh

set -e # Exit on error

echo "=========================================="
echo "UDP Face Detection Server Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
	echo -e "${RED}✗ Python 3.8+ required, found $PYTHON_VERSION${NC}"
	exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check required packages
echo ""
echo "Checking required packages..."
REQUIRED_PACKAGES=("cv2:opencv-python" "numpy:numpy" "sklearn:scikit-learn")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
	IFS=':' read -r import_name pip_name <<<"$pkg"
	if python3 -c "import $import_name" 2>/dev/null; then
		echo -e "${GREEN}✓ $pip_name installed${NC}"
	else
		echo -e "${YELLOW}✗ $pip_name missing${NC}"
		MISSING_PACKAGES+=("$pip_name")
	fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
	echo ""
	echo -e "${YELLOW}Installing missing packages...${NC}"
	pip3 install "${MISSING_PACKAGES[@]}"
	echo -e "${GREEN}✓ Packages installed${NC}"
fi

# Check project structure
echo ""
echo "Checking project structure..."

if [ ! -f "face_detector_cli.py" ]; then
	echo -e "${RED}✗ face_detector_cli.py not found${NC}"
	echo "Please run this script from the project root directory"
	exit 1
fi
echo -e "${GREEN}✓ face_detector_cli.py found${NC}"

# Check models
if [ ! -d "models" ]; then
	echo -e "${RED}✗ models directory not found${NC}"
	exit 1
fi

REQUIRED_MODELS=("svm_model.pkl" "bovw_encoder.pkl" "scaler.pkl")
for model in "${REQUIRED_MODELS[@]}"; do
	if [ ! -f "models/$model" ]; then
		echo -e "${RED}✗ models/$model not found${NC}"
		exit 1
	fi
	echo -e "${GREEN}✓ models/$model found${NC}"
done

# Create udp_server directory
echo ""
echo "Creating udp_server directory..."
mkdir -p udp_server
touch udp_server/__init__.py
echo -e "${GREEN}✓ udp_server directory created${NC}"

# Check if udp_server files exist
echo ""
echo "Checking UDP server files..."
REQUIRED_FILES=("config.py" "protocol.py" "frame_handler.py" "server.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
	if [ ! -f "udp_server/$file" ]; then
		MISSING_FILES+=("$file")
	else
		echo -e "${GREEN}✓ udp_server/$file found${NC}"
	fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
	echo ""
	echo -e "${YELLOW}Missing UDP server files:${NC}"
	for file in "${MISSING_FILES[@]}"; do
		echo "  - udp_server/$file"
	done
	echo ""
	echo "Please copy the following files to udp_server/:"
	echo "  - config.py"
	echo "  - protocol.py"
	echo "  - frame_handler.py"
	echo "  - server.py"
	exit 1
fi

# Check test client
echo ""
if [ -f "test_client.py" ]; then
	echo -e "${GREEN}✓ test_client.py found${NC}"
else
	echo -e "${YELLOW}⚠ test_client.py not found (optional)${NC}"
fi

# Check ports availability
echo ""
echo "Checking port availability..."
check_port() {
	if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":$1 "; then
		return 1
	else
		return 0
	fi
}

if check_port 5555; then
	echo -e "${GREEN}✓ Port 5555 available${NC}"
else
	echo -e "${YELLOW}⚠ Port 5555 is in use${NC}"
	echo "  You may need to stop the process using this port"
fi

if check_port 5556; then
	echo -e "${GREEN}✓ Port 5556 available${NC}"
else
	echo -e "${YELLOW}⚠ Port 5556 is in use${NC}"
	echo "  You may need to stop the process using this port"
fi

# Validation complete
echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "Quick start:"
echo ""
echo "1. Start the server:"
echo "   python3 -m udp_server.server --mode dev"
echo ""
echo "2. Test with image (in another terminal):"
echo "   python3 test_client.py image --image path/to/image.jpg"
echo ""
echo "3. Test with webcam:"
echo "   python3 test_client.py webcam --camera 0"
echo ""
echo "For more information:"
echo "  - Server setup: README_UDP_SERVER.md"
echo "  - Godot integration: GODOT_INTEGRATION.md"
echo "  - Quick start: QUICKSTART.md"
echo ""
echo "=========================================="
