# Forge Face Detection - InsightFace Edition

InsightFace worker for Forge. This service is intended to run as a separate TrueNAS SCALE app and exposes only `/health` and `/detect`.

## Features

- RetinaFace detection with ArcFace embeddings
- 512-dimensional face descriptors
- Optional age and gender estimates
- HEIC and EXIF orientation support
- Simple HTTP worker API for Forge

## Quick Start

### Recommended TrueNAS SCALE Deployment

1. Deploy this as its own TrueNAS app:
   - App Name: `forge-face-detection-insightface`
   - Image Repository: `ghcr.io/brwalknels/forge-face-detection-insightface`
   - Image Tag: `latest`
   - Host Port: `5001`
   - Container Port: `5001`

2. Mount the same photo dataset Forge uses:
   - Host path: `/mnt/pool/mastersync/private`
   - Container path: `/app/private`
   - Access: read-only

3. In Forge Admin -> Face Detection, set:
   - Worker URL: `http://YOUR-TRUENAS-IP:5001`
   - Background Processing Enabled: on
   - Auto-Process New Gallery Photos: on

No Docker reconnect step is required. Forge should talk to this worker over one stable URL.

## API Endpoints

### Health Check

```bash
GET http://YOUR-TRUENAS-IP:5001/health
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
POST http://YOUR-TRUENAS-IP:5001/detect
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
      "descriptor": [0.123, -0.456],
      "confidence": 0.98,
      "age": 25,
      "gender": "male"
    }
  ],
  "faceCount": 1,
  "processingTimeMs": 1523
}
```

## Notes

- Embedding size is `512`, so existing 128-dimensional face indexes from the older system must be rebuilt.
- The worker opens the exact `filePath` sent by Forge, so the shared photo dataset must be mounted at the same in-container path.
- Forge owns the queue, person records, labeling, and auto-assign logic. This worker only detects faces and returns embeddings.

## Environment Variables

- `MAX_IMAGE_SIZE`: Maximum image dimension before resize, default `2000`

## Building Locally

```bash
docker build -t ghcr.io/brwalknels/forge-face-detection-insightface:latest .
docker push ghcr.io/brwalknels/forge-face-detection-insightface:latest
```

## Troubleshooting

### Models Not Downloading

InsightFace downloads models on first run. This can take a minute or two. Check logs:

```bash
docker logs forge-face-detection-insightface
```

### High Memory Usage

InsightFace models are larger than the old face stack. Plan for roughly 2 GB of available RAM for comfortable operation.

### Slow Processing

This worker prioritizes accuracy over speed. Tune Forge's rate limit and processing schedule if you want gentler background usage.
