"""ONNX Runtime adapter for Ultralytics YOLO detection exports (YOLOv8 / YOLO11 / YOLO26, nms=False).

Expected export: input [1, 3, S, S] RGB float 0..1 (letterboxed), output [1, 4 + nc, N] with
(cx, cy, w, h) in input pixels followed by per-class scores. Class names and input size are read
from the model's own metadata and must match config/classes.*.yaml exactly; any mismatch is a hard
error (the system then reports NO REAL MODEL instead of guessing).
"""

import ast
from pathlib import Path

import cv2
import numpy as np

from app.perception.interface import ClassConfig, Detection

LETTERBOX_FILL = 114  # Ultralytics padding value


class ModelError(Exception):
    pass


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (img, scale, (pad_x, pad_y))."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


class OnnxDetector:
    vision_mode = "real_model"

    def __init__(self, weights: Path, classes: ClassConfig, conf_threshold: float, iou_threshold: float):
        import onnxruntime as ort

        if not weights.is_file():
            raise ModelError(f"model weights not found: {weights}")
        try:
            self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelError(f"ONNX Runtime could not load {weights.name}: {exc}") from exc
        meta = self.session.get_modelmeta().custom_metadata_map
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if len(inp.shape) != 4 or inp.shape[1] != 3 or not isinstance(inp.shape[2], int):
            raise ModelError(f"unsupported input shape {inp.shape}; expected fixed [1, 3, S, S]")
        self.input_size = int(inp.shape[2])
        if meta.get("task", "detect") != "detect":
            raise ModelError(f"model task is '{meta.get('task')}', only 'detect' is supported")
        if meta.get("end2end") == "True" or "'nms': True" in meta.get("args", ""):
            raise ModelError("end-to-end / NMS-embedded exports are not supported; export with nms=False")
        try:
            model_names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
        except Exception as exc:
            raise ModelError("model has no Ultralytics 'names' metadata; cannot verify classes") from exc
        if model_names != classes.model_classes:
            raise ModelError(
                f"model classes {model_names} do not match configured classes {classes.model_classes}. "
                "This model was not trained for this class map; refusing to run."
            )
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) != 3 or out_shape[1] != 4 + len(model_names):
            raise ModelError(f"unexpected output shape {out_shape}; expected [1, {4 + len(model_names)}, N]")
        self.classes = classes
        self.names = model_names
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.metadata = {
            "file": weights.name, "input_size": self.input_size, "classes": model_names,
            "description": meta.get("description"), "date": meta.get("date"), "license": meta.get("license"),
            "ultralytics_version": meta.get("version"),
        }

    def predict(self, frame_bgr: np.ndarray, frame_id: int) -> list[Detection]:
        img, r, (px, py) = letterbox(frame_bgr, self.input_size)
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (4 + nc, N)
        return self.decode(out, r, (px, py), frame_bgr.shape[1], frame_bgr.shape[0])

    def decode(self, out: np.ndarray, r: float, pad: tuple[int, int], width: int, height: int) -> list[Detection]:
        scores = out[4:]
        cls = scores.argmax(axis=0)
        conf = scores.max(axis=0)
        keep = conf >= self.conf
        if not np.any(keep):
            return []
        cx, cy, w, h = out[:4, keep]
        cls, conf = cls[keep], conf[keep]
        x1 = np.clip((cx - w / 2 - pad[0]) / r, 0, width)
        y1 = np.clip((cy - h / 2 - pad[1]) / r, 0, height)
        x2 = np.clip((cx + w / 2 - pad[0]) / r, 0, width)
        y2 = np.clip((cy + h / 2 - pad[1]) / r, 0, height)
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        idx = cv2.dnn.NMSBoxesBatched(boxes_xywh.tolist(), conf.tolist(), cls.tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(idx).flatten():
            name = self.names[int(cls[i])]
            dets.append(Detection(
                class_id=int(cls[i]), class_name=name, role=self.classes.role_of(name), confidence=float(conf[i]),
                bbox_xyxy_px=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            ))
        return dets
