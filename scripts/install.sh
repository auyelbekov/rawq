#!/bin/sh
# Install rawq — downloads the latest release binary and adds it to PATH.
# Usage: curl -fsSL https://raw.githubusercontent.com/auyelbekov/rawq/main/scripts/install.sh | sh

set -e

REPO="auyelbekov/rawq"
INSTALL_DIR="${RAWQ_INSTALL_DIR:-$HOME/.local/bin}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux*)   PLATFORM="linux" ;;
  Darwin*)  PLATFORM="macos" ;;
  *)        echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
  x86_64|amd64)  ARCH="x86_64" ;;
  aarch64|arm64)  ARCH="aarch64" ;;
  *)              echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

ARCHIVE="rawq-${PLATFORM}-${ARCH}.tar.gz"

# Get latest release tag
TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | head -1 | cut -d'"' -f4)
if [ -z "$TAG" ]; then
  echo "error: could not determine latest release"
  exit 1
fi

URL="https://github.com/$REPO/releases/download/$TAG/$ARCHIVE"

echo "Installing rawq $TAG for $PLATFORM-$ARCH..."
echo "  From: $URL"
echo "  To:   $INSTALL_DIR/rawq"

# Download and extract
mkdir -p "$INSTALL_DIR"
curl -fsSL "$URL" | tar xz -C "$INSTALL_DIR"
chmod +x "$INSTALL_DIR/rawq"

# Check PATH
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
  echo ""
  echo "Add to your PATH by adding this to your shell profile:"
  echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
fi

echo ""
echo "rawq $TAG installed successfully."
"$INSTALL_DIR/rawq" --version 2>/dev/null || true
