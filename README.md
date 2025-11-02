# Forge Face Detection - InsightFace Edition

Advanced face detection microservice using InsightFace (RetinaFace + ArcFace) for superior accuracy and fewer false positives compared to dlib.

## Features

- **RetinaFace Detector**: Industry-leading face detection with high accuracy and low false positive rate
- **ArcFace Recognition**: 512-dimensional face embeddings for superior face recognition
- **Better Quality**: Significantly reduces false positives (trees, sky, objects detected as faces)
- **Age & Gender**: Optional age and gender estimation
- **Production Ready**: Used in commercial face recognition systems worldwide

## Quick Start

### Docker Deployment (TrueNAS SCALE)

1. **Deploy via TrueNAS GUI:**
   - Go to Apps → Discover Apps → Custom App
   - App Name: `forge-face-detection-insightface`
   - Image Repository: `ghcr.io/brwalknels/forge-face-detection-insightface`
   - Image Tag: `latest`
   - Port: `5001` (host) → `5001` (container)

2. **Connect to Forge network:**
   ```bash
   docker network connect forge-network forge-face-detection-insightface
   ```

3. **Configure in Forge Admin:**
   - Go to Admin → Face Detection
   - Change Detection Model to "InsightFace (Accurate, Slow)"
   - Save settings

## API Endpoints

### Health Check
```bash
GET http://forge-face-detection-insightface:5001/health
```

Response:
```json
{
  "status": "ready",
  "service": "forge-face-detection-insightface",
  "model": "RetinaFace + ArcFace",
  "version": "2.0.0",
  "embedding_size": 512,
  "confidence_scores": true
}
```

### Detect Faces
```bash
POST http://forge-face-detection-insightface:5001/detect
Content-Type: application/json

{
  "fileId": "uuid",
  "filePath": "/app/private/user-id/photo.jpg"
}
```

Response:
```json
{
  "fileId": "uuid",
  "faces": [
    {
      "id": "uuid-face-0",
      "box": {
        "top": 100,
        "right": 300,
        "bottom": 250,
        "left": 150,
        "width": 150,
        "height": 150
      },
      "descriptor": [0.123, -0.456, ...],  // 512-dimensional
      "landmarks": {
        "left_eye": [180, 130],
        "right_eye": [220, 130],
        "nose": [200, 160],
        "mouth_left": [185, 190],
        "mouth_right": [215, 190]
      },
      "confidence": 0.98,
      "age": 25,
      "gender": "male"
    }
  ],
  "faceCount": 1,
  "processingTimeMs": 1523
}
```

## Performance

- **Processing Time**: 2-5 seconds per photo (CNN model: 1-3 seconds)
- **Accuracy**: 99%+ face detection accuracy (vs 95% for dlib)
- **False Positives**: <1% (vs 5-10% for dlib)
- **Embedding Size**: 512-dim (vs 128-dim for dlib)
- **Memory**: ~1.5GB RAM per worker (vs ~500MB for dlib)

## Model Comparison

| Feature | dlib CNN | InsightFace |
|---------|----------|-------------|
| Detection Model | CNN | RetinaFace |
| Recognition Model | ResNet | ArcFace |
| Embedding Size | 128-dim | 512-dim |
| False Positives | 5-10% | <1% |
| Processing Time | 1-3s | 2-5s |
| Memory | 500MB | 1.5GB |
| Age/Gender | ❌ | ✅ |
| Production Use | Research | Commercial |

## Environment Variables

- `MAX_IMAGE_SIZE`: Maximum image dimension (default: 2000px)

## Building Locally

```bash
# Build image
docker build -t ghcr.io/brwalknels/forge-face-detection-insightface:latest .

# Push to registry
docker push ghcr.io/brwalknels/forge-face-detection-insightface:latest
```

## Troubleshooting

### Models Not Downloading
InsightFace downloads models (~300MB) on first run. This takes 1-2 minutes. Check logs:
```bash
docker logs forge-face-detection-insightface
```

### High Memory Usage
InsightFace models are larger than dlib. Ensure at least 2GB RAM available per worker.

### Slow Processing
InsightFace prioritizes accuracy over speed. For faster processing, use dlib CNN model.

## License

InsightFace is licensed under MIT License.
