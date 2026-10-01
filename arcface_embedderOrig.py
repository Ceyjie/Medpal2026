"""
arcface_embedder.py -- ArcFace face embedding with landmark alignment.

Matches the interface of the existing FaceEmbedder:
    embedder.embed(face_crop) -> np.ndarray (512-D, L2-normalized)

Optionally accepts landmarks for proper alignment:
    embedder.embed(face_crop, landmarks=landmarks)

Landmarks MUST be in face_crop coordinates (not the parent frame).
If your landmarks come from SCRFD (which returns them in frame coords),
subtract (x1, y1) of the crop before calling embed().
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
    # skimage 0.26+ deprecates .estimate() in favor of from_estimate()
    try:
        transform = SimilarityTransform.from_estimate(landmark, alignment)
    except AttributeError:
        transform = SimilarityTransform()
        transform.estimate(landmark, alignment)
    matrix = transform.params[0:2, :]
    inverse = np.linalg.inv(transform.params)[0:2, :]
    return matrix, inverse


def align_face(image, landmark, image_size=112):
    """
    Warp face so landmarks match the ArcFace reference pose.
    Landmarks must be in `image` coordinates (not the parent frame).
    """
    M, M_inv = _estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size),
                            borderValue=0.0)
    return warped, M_inv


class ArcFaceEmbedder:
    def __init__(self, model_path, num_threads=4):
        """
        Args:
            model_path:  Path to the ArcFace ONNX model.
            num_threads: ONNX Runtime intra-op thread count. Set lower
                         (e.g. 3) when running training in a subprocess
                         so the tracker stays responsive. Set to 4 for
                         standalone scan-time use on a Pi 5.
        """
        self.input_size = (112, 112)
        so = ort.SessionOptions()
        so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        so.intra_op_num_threads = int(num_threads)
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        out_shape = self.session.get_outputs()[0].shape
        self.embedding_dim = (int(out_shape[1])
                              if len(out_shape) > 1 else 512)
        print(f"ArcFaceEmbedder: loaded {model_path}")
        print(f"  embedding dim: {self.embedding_dim}")
        print(f"  threads: {num_threads}")

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
        aligned to the ArcFace reference pose.

        Args:
            face_crop: BGR face image
            landmarks: optional (5, 2) array of keypoint coordinates
                       in the SAME COORDINATE SYSTEM as face_crop.
                       If they come from SCRFD (frame coords), subtract
                       (x1, y1) of the crop before calling.

        Returns:
            L2-normalized 512-D embedding, or None if input invalid
            or alignment failed.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        if landmarks is not None:
            h, w = face_crop.shape[:2]
            lm_max_x = float(landmarks[:, 0].max())
            lm_max_y = float(landmarks[:, 1].max())
            if lm_max_x > w * 2.0 or lm_max_y > h * 2.0:
                raise ValueError(
                    f"landmarks look like frame coords, not crop coords: "
                    f"max=({lm_max_x:.0f},{lm_max_y:.0f}), crop={w}x{h}. "
                    f"Subtract (x1, y1) before calling embed()."
                )
            try:
                aligned, _ = align_face(face_crop, landmarks)
            except Exception as e:
                print(f"ArcFace alignment failed: {e}")
                return None
        else:
            aligned = face_crop

        blob = self._preprocess(aligned)
        out = self.session.run(self.output_names,
                               {self.input_name: blob})[0]
        vec = out.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
