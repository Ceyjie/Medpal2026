"""
scrfd_detector.py -- SCRFD face detector with 5-point landmarks.

Returns detections as tuples:
    (x, y, w, h, conf, landmarks)
where landmarks is a (5, 2) numpy array of (x, y) points for
left-eye, right-eye, nose, left-mouth-corner, right-mouth-corner,
in FULL-FRAME coordinates.

Robust to both common SCRFD ONNX export layouts:
    - 3-D outputs:  (batch, anchors, channels)
    - 2-D outputs:  (anchors, channels)

Usage:
    detector = SCRFDDetector(config.SCRFD_MODEL_PATH,
                             input_size=config.SCRFD_INPUT_SIZE,
                             conf_thres=config.SCRFD_CONF_THRES,
                             iou_thres=config.SCRFD_IOU_THRES)
    detections = detector.infer(frame)
"""
import cv2
import numpy as np
import onnxruntime as ort


# SCRFD model parameters (standard for det_500m / det_2.5g / det_10g)
FMC = 3
FEAT_STRIDE_FPN = [8, 16, 32]
NUM_ANCHORS = 2
USE_KPS = True
MEAN = 127.5
STD = 128.0


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets, iou_thres):
    x1 = dets[:, 0]; y1 = dets[:, 1]
    x2 = dets[:, 2]; y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        indices = np.where(ovr <= iou_thres)[0]
        order = order[indices + 1]
    return keep


class SCRFDDetector:
    def __init__(self, model_path,
                 input_size=(640, 640),
                 conf_thres=0.5,
                 iou_thres=0.4):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.center_cache = {}

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path, sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_names = [i.name for i in self.session.get_inputs()]

        print(f"SCRFDDetector: loaded {model_path}")
        print(f"  input_size={input_size}, outputs={len(self.output_names)}")
        self._index_outputs()

    def _index_outputs(self):
        """
        Map (kind, stride) -> output index by running one dummy inference
        to materialize concrete shapes, then classifying by last dim
        (1 = score, 4 = bbox, 10 = kps) and sorting by anchor count.

        Handles both 3-D (batch, anchors, channels) and 2-D (anchors,
        channels) output layouts.
        """
        H, W = self.input_size[1], self.input_size[0]
        dummy = np.zeros((1, 3, H, W), dtype=np.float32)
        try:
            raw = self.session.run(self.output_names,
                                   {self.input_names[0]: dummy})
        except Exception as e:
            raise RuntimeError(f"SCRFD probe inference failed: {e}")

        by_dim = {1: [], 4: [], 10: []}
        for i, arr in enumerate(raw):
            a = np.asarray(arr)
            if a.ndim == 3:          # (batch, anchors, channels)
                anchor_count = int(a.shape[1])
                last_dim = int(a.shape[2])
            elif a.ndim == 2:        # (anchors, channels)
                anchor_count = int(a.shape[0])
                last_dim = int(a.shape[1])
            else:
                continue
            if last_dim in by_dim:
                by_dim[last_dim].append((i, anchor_count))

        if any(len(by_dim[d]) != 3 for d in by_dim):
            shapes = [list(np.asarray(a).shape) for a in raw]
            raise RuntimeError(
                f"SCRFD: expected 3 score / 3 bbox / 3 kps outputs. "
                f"Got {len(by_dim[1])}/{len(by_dim[4])}/{len(by_dim[10])}. "
                f"Shapes: {shapes}"
            )

        self._out_idx = {}
        for kind, d in (("score", 1), ("bbox", 4), ("kps", 10)):
            entries = sorted(by_dim[d], key=lambda t: -t[1])
            for stride, (i, n) in zip(FEAT_STRIDE_FPN, entries):
                self._out_idx[(kind, stride)] = i
                print(f"    {kind:5s} stride={stride:2d}  idx={i}  "
                      f"anchors={n}")

    def _forward(self, image, threshold):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(image.shape[0:2][::-1])   # (w, h)
        blob = cv2.dnn.blobFromImage(
            image, 1.0 / STD, input_size, (MEAN, MEAN, MEAN), swapRB=True,
        )
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for stride in FEAT_STRIDE_FPN:
            scores = outputs[self._out_idx[("score", stride)]]
            bbox_preds = outputs[self._out_idx[("bbox", stride)]] * stride
            kps_preds = (outputs[self._out_idx[("kps", stride)]] * stride
                         if USE_KPS else None)

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if NUM_ANCHORS > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * NUM_ANCHORS, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            pos_inds = np.where(scores >= threshold)[0]

            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if USE_KPS:
                kps_preds = kps_preds.reshape(-1, 10)
                kpss = _distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def infer(self, frame):
        """
        Detect faces in a BGR frame.

        Returns a list of tuples:
            (x, y, w, h, conf, landmarks)
        where landmarks is a (5, 2) array of (x, y) points in
        FULL-FRAME coordinates:
            [left-eye, right-eye, nose, left-mouth, right-mouth]
        """
        width, height = self.input_size
        im_ratio = float(frame.shape[0]) / frame.shape[1]
        model_ratio = height / width

        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / frame.shape[0]
        resized = cv2.resize(frame, (new_width, new_height))
        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized

        scores_list, bboxes_list, kpss_list = self._forward(
            det_image, self.conf_thres
        )

        # --- FIX: scores are 1-D per stride; concatenate, don't vstack ---
        if not scores_list:
            return []
        scores = np.concatenate(scores_list) if scores_list else np.array([])
        if scores.size == 0:
            return []

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(
            np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = _nms(pre_det, self.iou_thres)
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        results = []
        for i in range(det.shape[0]):
            x1, y1, x2, y2, score = det[i]
            x = max(0, int(x1))
            y = max(0, int(y1))
            w = max(1, int(x2 - x1))
            h = max(1, int(y2 - y1))
            results.append((x, y, w, h, float(score), kpss[i]))

        return results

    def top_detection_label(self):
        return "SCRFD"
