#!/bin/bash
# Quick push script for forge-face-detection-insightface

echo "Building and pushing forge-face-detection-insightface..."

# Build image
docker build -t ghcr.io/brwalknels/forge-face-detection-insightface:latest .

if [ $? -eq 0 ]; then
    echo "Build successful! Pushing to registry..."
    docker push ghcr.io/brwalknels/forge-face-detection-insightface:latest
    
    if [ $? -eq 0 ]; then
        echo "✓ Push successful!"
        echo "Image: ghcr.io/brwalknels/forge-face-detection-insightface:latest"
    else
        echo "✗ Push failed!"
        exit 1
    fi
else
    echo "✗ Build failed!"
    exit 1
fi
