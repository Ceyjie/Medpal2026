"""
arcface_embedder.py -- ArcFace face embedding with landmark alignment.
"""
import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform


REFERENCE_ALIGNMENT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _estimate_norm(landmark, image_size=112):
    if landmark.shape != (5, 2):
        raise ValueError(f"landmark must be (5, 2), got {landmark.shape}")
    ratio = float(image_size) / 112.0
    alignment = REFERENCE_ALIGNMENT * ratio
    try:
        transform = SimilarityTransform.from_estimate(landmark, alignment)
    except AttributeError:
        transform = SimilarityTransform()
        transform.estimate(landmark, alignment)
    matrix = transform.params[0:2, :]
    inverse = np.linalg.inv(transform.params)[0:2, :]
    return matrix, inverse


def align_face(image, landmark, image_size=112):
    M, M_inv = _estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size),
                            borderValue=0.0)
    return warped, M_inv


class ArcFaceEmbedder:
    def __init__(self, model_path, num_threads=4):
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
        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob

    def embed(self, face_crop, landmarks=None):
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
