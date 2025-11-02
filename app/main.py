"""
Forge Face Detection Microservice - InsightFace Edition
Python Flask API for facial recognition using InsightFace (RetinaFace + ArcFace)

Better accuracy, fewer false positives compared to dlib.
"""
import os
import logging
from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis
from insightface.data import get_image
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import time
from pathlib import Path

# Register HEIF/HEIC support for PIL
register_heif_opener()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from environment variables
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2000'))  # Max dimension for processing
DETECTION_SIZE = (640, 640)  # RetinaFace input size

logger.info(f"InsightFace Detection Service Starting...")
logger.info(f"Max Image Size: {MAX_IMAGE_SIZE}px")
logger.info(f"Detection Size: {DETECTION_SIZE}")

# Initialize InsightFace
# det_name='retinaface_r50_v1' - RetinaFace detector (more accurate than dlib)
# rec_name='arcface_r100_v1' - ArcFace recognition model (512-dim embeddings)
face_app = None

def initialize_face_app():
    """Initialize InsightFace FaceAnalysis with RetinaFace + ArcFace"""
    global face_app
    try:
        logger.info("Initializing InsightFace models...")
        face_app = FaceAnalysis(
            name='buffalo_l',  # Large model pack (includes RetinaFace + ArcFace)
            providers=['CPUExecutionProvider']  # Use CPU (add CUDAExecutionProvider for GPU)
        )
        # det_size: detection resolution (640x640 - good balance of speed/accuracy)
        # det_thresh: minimum detection score (0.5 default, higher = fewer false positives)
        face_app.prepare(ctx_id=0, det_size=DETECTION_SIZE, det_thresh=0.5)
        logger.info("✓ InsightFace initialized successfully (RetinaFace + ArcFace)")
        logger.info(f"Detection threshold: 0.5 (adjustable via API)")
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}", exc_info=True)
        raise

# Initialize on startup
initialize_face_app()


def load_and_preprocess_image(file_path):
    """
    Load image and convert to format expected by InsightFace (BGR numpy array)
    
    Args:
        file_path: Path to image file
        
    Returns:
        numpy.ndarray: BGR image ready for InsightFace
    """
    try:
        # Load with PIL (handles HEIC/HEIF)
        pil_image = Image.open(file_path)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize if too large
        width, height = pil_image.size
        if max(width, height) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image: {width}x{height} → {new_width}x{new_height}")
        
        # Convert to numpy array (RGB)
        image_rgb = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV/InsightFace expects BGR)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for Docker and monitoring
    Returns service status and configuration
    """
    return jsonify({
        'status': 'ready' if face_app is not None else 'initializing',
        'service': 'forge-face-detection-insightface',
        'model': 'RetinaFace + ArcFace',
        'version': '2.0.0',
        'embedding_size': 512,
        'confidence_scores': True
    }), 200


@app.route('/detect', methods=['POST'])
def detect_faces():
    """
    Detect faces in a photo using InsightFace and return face descriptors
    
    Request Body (JSON):
    {
        "fileId": "uuid-of-file",
        "filePath": "/app/private/user-id/photo.jpg"
    }
    
    Response:
    {
        "fileId": "uuid-of-file",
        "faces": [
            {
                "id": "face-uuid",
                "box": {"top": 100, "right": 300, "bottom": 250, "left": 150},
                "descriptor": [0.123, -0.456, ...],  # 512-dimensional vector (ArcFace)
                "landmarks": {...},  # 5 facial landmark points
                "confidence": 0.95,  # Detection confidence
                "age": 25,           # Estimated age
                "gender": "male"     # Estimated gender
            }
        ],
        "faceCount": 2,
        "processingTimeMs": 1523
    }
    """
    start_time = time.time()
    
    if face_app is None:
        return jsonify({'error': 'Face detection service not initialized'}), 503
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        file_id = data.get('fileId')
        file_path = data.get('filePath')
        
        if not file_id or not file_path:
            return jsonify({'error': 'Missing required fields: fileId, filePath'}), 400
        
        # Verify file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return jsonify({'error': f'File not found: {file_path}'}), 404
        
        logger.info(f"Processing {file_id}: {Path(file_path).name}")
        
        # Load and preprocess image
        try:
            image = load_and_preprocess_image(file_path)
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {str(e)}")
            return jsonify({'error': f'Failed to load image: {str(e)}'}), 400
        
        # Get confidence threshold from request (default 0.5 for InsightFace - stricter than dlib)
        min_confidence = data.get('min_confidence', 0.5)
        logger.info(f"Using confidence threshold: {min_confidence}")
        
        # Detect faces with InsightFace
        detected_faces = face_app.get(image)
        
        # Log all detected faces with scores for debugging
        if detected_faces:
            scores = [f"{face.det_score:.3f}" for face in detected_faces]
            logger.info(f"Detected {len(detected_faces)} face(s) with scores: {', '.join(scores)}")
        
        # Filter by confidence threshold
        if detected_faces:
            filtered_faces = [face for face in detected_faces if face.det_score >= min_confidence]
            filtered_count = len(detected_faces) - len(filtered_faces)
            if filtered_count > 0:
                filtered_scores = [f"{face.det_score:.3f}" for face in detected_faces if face.det_score < min_confidence]
                logger.info(f"Filtered {filtered_count} low-confidence detection(s): {', '.join(filtered_scores)} (threshold: {min_confidence})")
            detected_faces = filtered_faces
        
        if not detected_faces:
            logger.info(f"No faces detected in {file_id}")
            processing_time = int((time.time() - start_time) * 1000)
            return jsonify({
                'fileId': file_id,
                'faces': [],
                'faceCount': 0,
                'processingTimeMs': processing_time
            }), 200
        
        # Build response
        faces = []
        for i, face in enumerate(detected_faces):
            # Bounding box (x1, y1, x2, y2 format from InsightFace)
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Face landmarks (5 points: left_eye, right_eye, nose, mouth_left, mouth_right)
            landmarks = face.kps.astype(int) if face.kps is not None else None
            landmarks_dict = {}
            if landmarks is not None:
                landmarks_dict = {
                    'left_eye': (int(landmarks[0][0]), int(landmarks[0][1])),
                    'right_eye': (int(landmarks[1][0]), int(landmarks[1][1])),
                    'nose': (int(landmarks[2][0]), int(landmarks[2][1])),
                    'mouth_left': (int(landmarks[3][0]), int(landmarks[3][1])),
                    'mouth_right': (int(landmarks[4][0]), int(landmarks[4][1]))
                }
            
            # Face embedding (512-dimensional ArcFace vector)
            descriptor = face.normed_embedding.tolist()
            
            # Detection confidence
            confidence = float(face.det_score)
            
            # Age and gender (if available)
            age = int(face.age) if hasattr(face, 'age') and face.age is not None else None
            gender = 'male' if hasattr(face, 'gender') and face.gender == 1 else 'female' if hasattr(face, 'gender') and face.gender == 0 else None
            
            face_data = {
                'id': f"{file_id}-face-{i}",
                'box': {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left),
                    'width': int(right - left),
                    'height': int(bottom - top)
                },
                'descriptor': descriptor,  # 512-dim ArcFace embedding
                'landmarks': landmarks_dict,
                'confidence': round(confidence, 3)
            }
            
            # Add age/gender if available
            if age is not None:
                face_data['age'] = age
            if gender is not None:
                face_data['gender'] = gender
            
            faces.append(face_data)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"✓ Detected {len(faces)} face(s) in {file_id} ({processing_time}ms)")
        
        return jsonify({
            'fileId': file_id,
            'faces': faces,
            'faceCount': len(faces),
            'processingTimeMs': processing_time
        }), 200
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Face detection error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Face detection failed',
            'message': str(e),
            'processingTimeMs': processing_time
        }), 500


@app.route('/batch-detect', methods=['POST'])
def batch_detect_faces():
    """
    Detect faces in multiple photos (batch processing)
    
    Request Body (JSON):
    {
        "photos": [
            {"fileId": "uuid-1", "filePath": "/app/private/user/photo1.jpg"},
            {"fileId": "uuid-2", "filePath": "/app/private/user/photo2.jpg"}
        ]
    }
    
    Response:
    {
        "results": [
            {"fileId": "uuid-1", "faces": [...], "faceCount": 2},
            {"fileId": "uuid-2", "faces": [...], "faceCount": 1}
        ],
        "totalPhotos": 2,
        "totalFaces": 3,
        "processingTimeMs": 3500
    }
    """
    start_time = time.time()
    
    if face_app is None:
        return jsonify({'error': 'Face detection service not initialized'}), 503
    
    try:
        data = request.get_json()
        photos = data.get('photos', [])
        
        if not photos:
            return jsonify({'error': 'No photos provided'}), 400
        
        results = []
        total_faces = 0
        
        for photo in photos:
            file_id = photo.get('fileId')
            file_path = photo.get('filePath')
            
            if not file_id or not file_path:
                results.append({
                    'fileId': file_id or 'unknown',
                    'error': 'Missing fileId or filePath',
                    'faces': [],
                    'faceCount': 0
                })
                continue
            
            # Process each photo
            try:
                if not os.path.exists(file_path):
                    results.append({
                        'fileId': file_id,
                        'error': 'File not found',
                        'faces': [],
                        'faceCount': 0
                    })
                    continue
                
                image = load_and_preprocess_image(file_path)
                detected_faces = face_app.get(image)
                
                faces = []
                for i, face in enumerate(detected_faces):
                    bbox = face.bbox.astype(int)
                    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    landmarks = face.kps.astype(int) if face.kps is not None else None
                    landmarks_dict = {}
                    if landmarks is not None:
                        landmarks_dict = {
                            'left_eye': (int(landmarks[0][0]), int(landmarks[0][1])),
                            'right_eye': (int(landmarks[1][0]), int(landmarks[1][1])),
                            'nose': (int(landmarks[2][0]), int(landmarks[2][1])),
                            'mouth_left': (int(landmarks[3][0]), int(landmarks[3][1])),
                            'mouth_right': (int(landmarks[4][0]), int(landmarks[4][1]))
                        }
                    
                    descriptor = face.normed_embedding.tolist()
                    confidence = float(face.det_score)
                    
                    face_data = {
                        'id': f"{file_id}-face-{i}",
                        'box': {
                            'top': int(top),
                            'right': int(right),
                            'bottom': int(bottom),
                            'left': int(left),
                            'width': int(right - left),
                            'height': int(bottom - top)
                        },
                        'descriptor': descriptor,
                        'landmarks': landmarks_dict,
                        'confidence': round(confidence, 3)
                    }
                    
                    # Add age/gender if available
                    if hasattr(face, 'age') and face.age is not None:
                        face_data['age'] = int(face.age)
                    if hasattr(face, 'gender') and face.gender is not None:
                        face_data['gender'] = 'male' if face.gender == 1 else 'female'
                    
                    faces.append(face_data)
                
                results.append({
                    'fileId': file_id,
                    'faces': faces,
                    'faceCount': len(faces)
                })
                
                total_faces += len(faces)
                
            except Exception as e:
                logger.error(f"Error processing {file_id}: {str(e)}")
                results.append({
                    'fileId': file_id,
                    'error': str(e),
                    'faces': [],
                    'faceCount': 0
                })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"✓ Batch processed {len(photos)} photos, found {total_faces} faces ({processing_time}ms)")
        
        return jsonify({
            'results': results,
            'totalPhotos': len(photos),
            'totalFaces': total_faces,
            'processingTimeMs': processing_time
        }), 200
        
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Batch detection failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Development server (use gunicorn in production)
    app.run(host='0.0.0.0', port=5001, debug=False)
