# Quick push script for forge-face-detection-insightface (Windows PowerShell)

Write-Host "Building and pushing forge-face-detection-insightface..." -ForegroundColor Cyan

# Build image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
docker build -t ghcr.io/brwalknels/forge-face-detection-insightface:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Build successful! Pushing to registry..." -ForegroundColor Green
    docker push ghcr.io/brwalknels/forge-face-detection-insightface:latest
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Push successful!" -ForegroundColor Green
        Write-Host "Image: ghcr.io/brwalknels/forge-face-detection-insightface:latest" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ Push failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n✗ Build failed!" -ForegroundColor Red
    exit 1
}
