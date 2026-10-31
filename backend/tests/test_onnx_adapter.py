"""M6: ONNX adapter. Synthetic Ultralytics-style exports make decoding deterministic."""

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from app.perception.interface import ClassConfig, load_class_config  # noqa: E402
from app.perception.onnx_adapter import ModelError, OnnxDetector, letterbox  # noqa: E402
from saltsort_common.paths import REPO_ROOT  # noqa: E402

SALT_CLASSES = {0: "impure", 1: "pure", 2: "unwanted"}
CLASS_CFG = ClassConfig(model_classes=SALT_CLASSES,
                        roles={"reject": ["impure"], "accept": ["pure"], "obstruction": ["unwanted"]})


def make_model(path: Path, output: np.ndarray, names: dict, size: int = 320) -> Path:
    """Model whose output is a fixed tensor (input is consumed via 0 * ReduceSum so it stays live)."""
    const = numpy_helper.from_array(output.astype(np.float32), "fixed")
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["fixed", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "synthetic",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(output.shape))],
        initializer=[const, numpy_helper.from_array(np.array(0.0, np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for k, v in {"names": str(names), "task": "detect", "imgsz": str([size, size]), "stride": "32",
                 "args": "{'nms': False}", "description": "synthetic test model"}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, path)
    return path


def predictions(boxes_scores: list[tuple], nc: int = 3, n: int = 20) -> np.ndarray:
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(boxes_scores):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


def test_letterbox_geometry():
    img = np.zeros((480, 640, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert r == pytest.approx(0.5) and (px, py) == (0, 40)


def test_decode_maps_letterboxed_boxes_back_to_image(tmp_path):
    # 640x480 frame -> scale 0.5, pad_y 40. A box centered at (160, 140) in model space is (320, 200)
    # in the image; 40x40 model px -> 80x80 image px.
    out = predictions([(160, 140, 40, 40, 0, 0.9), (161, 141, 40, 40, 0, 0.8),  # duplicate -> NMS
                       (250, 200, 30, 30, 1, 0.95), (50, 50, 10, 10, 2, 0.3)])  # last below threshold
    det = OnnxDetector(make_model(tmp_path / "m.onnx", out, SALT_CLASSES), CLASS_CFG, 0.5, 0.45)
    dets = det.predict(np.zeros((480, 640, 3), np.uint8), 1)
    assert len(dets) == 2
    impure = next(d for d in dets if d.class_name == "impure")
    assert impure.role == "reject" and impure.confidence == pytest.approx(0.9)
    assert impure.bbox_xyxy_px == pytest.approx((280, 160, 360, 240), abs=1e-3)
    pure = next(d for d in dets if d.class_name == "pure")
    assert pure.role == "accept"


def test_class_mismatch_is_refused(tmp_path):
    out = predictions([], nc=1)
    path = make_model(tmp_path / "bird.onnx", out, {0: "bird"}, size=960)
    with pytest.raises(ModelError, match="do not match"):
        OnnxDetector(path, CLASS_CFG, 0.5, 0.45)


def test_missing_weights_and_garbage_file(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        OnnxDetector(tmp_path / "nope.onnx", CLASS_CFG, 0.5, 0.45)
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelError):
        OnnxDetector(bad, CLASS_CFG, 0.5, 0.45)


def test_example_class_config_is_consistent():
    cfg = load_class_config(REPO_ROOT / "config" / "classes.example.yaml")
    assert cfg.model_classes == SALT_CLASSES
    assert cfg.role_of("unwanted") == "obstruction"


SUPPLIED = REPO_ROOT.parent / "best.onnx"


@pytest.mark.skipif(not SUPPLIED.is_file(), reason="supplied model not present next to the repo")
def test_supplied_bird_model_is_refused_for_salt_classes():
    with pytest.raises(ModelError, match="do not match"):
        OnnxDetector(SUPPLIED, CLASS_CFG, 0.5, 0.45)
