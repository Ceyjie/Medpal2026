"""
arcface_embedder.py -- ArcFace face embedding with landmark alignment.

Matches the interface of the existing FaceEmbedder:
    embedder.embed(face_crop) -> np.ndarray (512-D, L2-normalized)

Optionally accepts landmarks for proper alignment:
    embedder.embed(face_crop, landmarks=landmarks)
"""
import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform


# Reference alignment for ArcFace 112x112 output
REFERENCE_ALIGNMENT = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def _estimate_norm(landmark, image_size=112):
    """Compute 2x3 affine matrix that aligns landmark to reference."""
    if landmark.shape != (5, 2):
        raise ValueError(f"landmark must be (5, 2), got {landmark.shape}")
    ratio = float(image_size) / 112.0
    alignment = REFERENCE_ALIGNMENT * ratio
    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)
    matrix = transform.params[0:2, :]
    inverse = np.linalg.inv(transform.params)[0:2, :]
    return matrix, inverse


def align_face(image, landmark, image_size=112):
    """Warp face so landmarks match the ArcFace reference pose."""
    M, M_inv = _estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size),
                            borderValue=0.0)
    return warped, M_inv


class ArcFaceEmbedder:
    def __init__(self, model_path):
        self.input_size = (112, 112)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.embedding_dim = self.session.get_outputs()[0].shape[1]
        print(f"ArcFaceEmbedder: loaded {model_path}")
        print(f"  embedding dim: {self.embedding_dim}")

    def _preprocess(self, aligned_face):
        """ArcFace normalization: [-1, 1] on 112x112 RGB."""
        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob

    def embed(self, face_crop, landmarks=None):
        """
        Embed a face crop. If landmarks are provided, the face is first
        aligned to the ArcFace reference pose, which improves accuracy.

        Args:
            face_crop: BGR face image
            landmarks: optional (5, 2) array of keypoint coordinates
                       in the same coordinate system as face_crop

        Returns:
            L2-normalized 512-D embedding, or None if input is invalid.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        if landmarks is not None:
            # Full pipeline: alignment + embedding
            try:
                aligned, _ = align_face(face_crop, landmarks)
            except Exception as e:
                print(f"ArcFace alignment failed: {e}")
                aligned = face_crop
        else:
            # Fallback: just resize (lower accuracy)
            aligned = face_crop

        blob = self._preprocess(aligned)
        out = self.session.run(self.output_names,
                               {self.input_name: blob})[0]
        vec = out.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec