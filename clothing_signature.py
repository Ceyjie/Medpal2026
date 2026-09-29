"""
clothing_signature.py -- Lightweight appearance signature for a person.

Extracts a fixed-length feature vector from a body crop that captures
clothing colour and texture. Used to maintain a person's identity across
frames where the face is not visible (turned away, back to camera).

Runs on CPU in ~3-5 ms per crop. No model, no GPU, no Coral.

Typical use:
    sig = extract_clothing_signature(body_crop)
    score = compare_signatures(sig_a, sig_b)   # 0..1, higher = same
"""
import cv2
import numpy as np


# Torso region: the top 55% of the body crop, excluding the head.
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.55

# Colour histogram bins (HSV)
H_BINS = 12
S_BINS = 8
V_BINS = 4

# HOG parameters.
# For winSize=(64,64), blockSize=(32,32), blockStride=(16,16),
# cellSize=(16,16), nbins=9, OpenCV produces 324 elements:
#   9 blocks * 4 cells/block * 9 bins = 324.
HOG_WIN = (64, 64)
HOG_CELL = (16, 16)
HOG_BLOCK = (2, 2)
HOG_BINS = 9


_hog = cv2.HOGDescriptor(
    _winSize=HOG_WIN,
    _blockSize=(HOG_CELL[0] * HOG_BLOCK[0],
                HOG_CELL[1] * HOG_BLOCK[1]),
    _blockStride=(HOG_CELL[0], HOG_CELL[1]),
    _cellSize=HOG_CELL,
    _nbins=HOG_BINS,
)


def _torso_crop(body_crop):
    """Return the torso sub-region of a body crop."""
    h, w = body_crop.shape[:2]
    y1 = int(h * TORSO_TOP)
    y2 = int(h * TORSO_BOTTOM)
    if y2 <= y1:
        return body_crop
    return body_crop[y1:y2, :]


def _color_histogram(torso_bgr):
    """HSV histogram over the torso, normalised to sum to 1."""
    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [H_BINS, S_BINS, V_BINS],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten()
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32)


def _hog_descriptor(torso_bgr):
    """HOG descriptor over the torso, resized to a fixed window."""
    gray = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, HOG_WIN)
    hog = _hog.compute(gray)
    if hog is None:
        n_cells_x = HOG_WIN[0] // HOG_CELL[0]
        n_cells_y = HOG_WIN[1] // HOG_CELL[1]
        n_blocks_x = n_cells_x - HOG_BLOCK[0] + 1
        n_blocks_y = n_cells_y - HOG_BLOCK[1] + 1
        return np.zeros(
            n_blocks_x * n_blocks_y *
            HOG_BLOCK[0] * HOG_BLOCK[1] * HOG_BINS,
            dtype=np.float32,
        )
    hog = hog.flatten()
    norm = np.linalg.norm(hog)
    if norm > 0:
        hog = hog / norm
    return hog.astype(np.float32)


def extract_clothing_signature(body_crop):
    """
    Extract a clothing signature from a BGR body crop.

    Returns a dict with:
        'color':     colour histogram (H_BINS*S_BINS*V_BINS floats)
        'hog':       HOG descriptor (324 floats on stock OpenCV)
        'combined':  concatenation of both, L2-normalised
    Returns None if the crop is empty or too small.
    """
    if body_crop is None or body_crop.size == 0:
        return None
    h, w = body_crop.shape[:2]
    if h < 20 or w < 10:
        return None

    torso = _torso_crop(body_crop)
    color = _color_histogram(torso)
    hog = _hog_descriptor(torso)

    combined = np.concatenate([color, hog]).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    return {"color": color, "hog": hog, "combined": combined}


def compare_signatures(sig_a, sig_b, color_weight=0.65):
    """
    Similarity between two signatures, 0..1.

    Colour is weighted heavily because it is the most stable cue for
    clothing. HOG catches texture and pattern.

    Zero-vector handling:
      - Both HOG descriptors near-zero (solid-colour clothing):
        use colour similarity only.
      - One near-zero, one not: don't trust HOG, use colour only.
      - Both meaningful: weighted combination of colour and HOG.
    """
    if sig_a is None or sig_b is None:
        return 0.0

    # Colour: Bhattacharyya coefficient (1.0 = identical distribution)
    color_score = float(np.sum(np.sqrt(sig_a["color"] * sig_b["color"])))

    hog_a = sig_a["hog"]
    hog_b = sig_b["hog"]
    na = float(np.linalg.norm(hog_a))
    nb = float(np.linalg.norm(hog_b))

    if na < 1e-6 or nb < 1e-6:
        # Either both flat or one is flat; HOG isn't informative here.
        return max(0.0, min(1.0, color_score))

    hog_score = float(np.dot(hog_a, hog_b) / (na * nb))
    score = color_weight * color_score + (1.0 - color_weight) * hog_score
    return max(0.0, min(1.0, score))


def update_reference_signature(old_sig, new_sig, alpha=0.10):
    """
    Exponential moving average of a reference signature.
    Call this each frame the person is confidently identified.
    """
    if old_sig is None:
        return new_sig
    if new_sig is None:
        return old_sig
    return {
        "color": (1 - alpha) * old_sig["color"]
                 + alpha * new_sig["color"],
        "hog":   (1 - alpha) * old_sig["hog"]
                 + alpha * new_sig["hog"],
        "combined": (
            (1 - alpha) * old_sig["combined"]
            + alpha * new_sig["combined"]
        ),
    }
