from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from blockpose.io import CameraModel
from blockpose.rgb_annotation import (
    AsyncSegmentationWorker,
    MaskSelectionParams,
    TransformersSam2Backend,
    build_projection_cache,
    candidate_mask_at,
    load_candidate_masks,
    map_widget_to_image,
    segment_with_backend,
    select_points_from_mask,
)


def camera() -> CameraModel:
    return CameraModel(np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]]), 100, 80)


def test_aspect_mapping_accounts_for_letterbox_and_edges() -> None:
    assert map_widget_to_image(100, 50, 200, 100, 100, 100) == (50, 50)
    assert map_widget_to_image(49, 50, 200, 100, 100, 100) is None
    assert map_widget_to_image(149.9, 99.9, 200, 100, 100, 100) == (99, 99)
    assert map_widget_to_image(np.nan, 0, 10, 10, 10, 10) is None


def test_projection_cache_marks_front_and_bounds() -> None:
    points = np.array([[0, 0, 2], [2, 0, 1], [0, 0, -1]], dtype=float)
    cache = build_projection_cache(points, np.eye(4), camera())
    np.testing.assert_array_equal(cache.valid, [True, False, False])
    np.testing.assert_allclose(cache.uv[0], [50, 40])


def test_mask_selection_rejects_background_depth_and_disconnected_clutter() -> None:
    # Dense front surface, a background copy projecting to the same region, and
    # a spatially disconnected same-depth patch inside the broad 2D mask.
    xy = np.array([(x, y) for x in np.linspace(-0.2, 0.2, 9) for y in np.linspace(-0.2, 0.2, 9)])
    front = np.column_stack([xy, np.full(len(xy), 2.0)])
    back = front * 2.0
    remote = front + np.array([0.9, 0.0, 0.0])
    points = np.vstack([front, back, remote])
    cache = build_projection_cache(points, np.eye(4), camera())
    mask = np.ones((80, 100), dtype=bool)
    result = select_points_from_mask(
        points,
        cache,
        mask,
        (50, 40),
        MaskSelectionParams(
            dilation_px=0,
            depth_tolerance_m=0.3,
            spatial_radius_m=1.5,
            connectivity_radius_m=0.09,
            min_points=20,
        ),
    )
    assert result.success
    assert len(result.indices) == len(front)
    assert np.all(result.indices < len(front))


def test_selection_failure_is_structured_and_non_throwing() -> None:
    points = np.array([[0, 0, 2]], dtype=float)
    cache = build_projection_cache(points, np.eye(4), camera())
    result = select_points_from_mask(points, cache, np.zeros((80, 100), bool), (50, 40))
    assert not result.success
    assert result.points.shape == (0, 3)
    assert "too few" in result.reason
    wrong_shape = select_points_from_mask(points, cache, np.zeros((2, 2), bool), (0, 0))
    assert not wrong_shape.success
    assert "does not match" in wrong_shape.reason


def test_candidate_choice_prefers_smallest_overlapping_mask() -> None:
    broad = np.ones((5, 5), bool)
    small = np.zeros((5, 5), bool)
    small[2, 2] = True
    selected = candidate_mask_at({1: broad, 2: small}, (2, 2))
    assert selected is not None and selected[0] == 2
    assert candidate_mask_at({1: broad}, (10, 10)) is None


def test_candidate_loader_ignores_bad_shape_missing_and_path_escape(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    valid = np.zeros((8, 10), dtype=np.uint8)
    valid[2:4, 3:5] = 255
    plt.imsave(snapshot / "valid.png", valid, cmap="gray", vmin=0, vmax=255)
    plt.imsave(snapshot / "wrong.png", np.zeros((2, 2)), cmap="gray")
    (snapshot / "metadata.yaml").write_text(
        "candidate_masks:\n"
        "  - {detection_id: 1, file: valid.png}\n"
        "  - {detection_id: 2, file: wrong.png}\n"
        "  - {detection_id: 3, file: ../escape.png}\n"
        "  - {detection_id: nope, file: missing.png}\n"
    )
    loaded = load_candidate_masks(snapshot, (8, 10))
    assert set(loaded) == {1}
    assert loaded[1][2, 3]


def test_sam_backend_is_lazy() -> None:
    backend = TransformersSam2Backend("does-not-load-during-construction")
    assert backend._model is None
    assert backend._processor is None


def test_fake_backend_contract_and_shape_validation() -> None:
    class FakeBackend:
        def segment(self, rgb: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
            result = np.zeros(rgb.shape[:2], bool)
            result[point_xy[1], point_xy[0]] = True
            return result

    rgb = np.zeros((6, 8, 3), np.uint8)
    result = segment_with_backend(FakeBackend(), rgb, (3, 2))
    assert result[2, 3]

    class BadBackend:
        def segment(self, rgb: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
            return np.zeros((1, 1), bool)

    with np.testing.assert_raises_regex(ValueError, "does not match"):
        segment_with_backend(BadBackend(), rgb, (3, 2))


class _SlowBackend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def segment(self, rgb: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
        self.calls += 1
        self.started.set()
        assert self.release.wait(2)
        return np.zeros(rgb.shape[:2], bool)


def test_async_worker_is_single_flight_and_delivers_once() -> None:
    backend = _SlowBackend()
    worker = AsyncSegmentationWorker(backend)
    scheduled: list = []
    completed: list = []
    image = np.zeros((4, 5, 3), np.uint8)
    assert worker.submit(image, (1, 1), "a", scheduled.append, lambda *args: completed.append(args))
    assert backend.started.wait(1)
    assert not worker.submit(
        image, (2, 2), "b", scheduled.append, lambda *args: completed.append(args)
    )
    backend.release.set()
    deadline = time.monotonic() + 1
    while not scheduled and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(scheduled) == 1
    assert worker.busy
    scheduled.pop()()
    assert not worker.busy
    assert len(completed) == 1 and completed[0][2] == "a"
    assert backend.calls == 1
    worker.close()


def test_async_worker_suppresses_stale_and_closed_results() -> None:
    image = np.zeros((4, 5, 3), np.uint8)
    for action in ("invalidate", "close"):
        backend = _SlowBackend()
        worker = AsyncSegmentationWorker(backend)
        scheduled: list = []
        assert worker.submit(image, (1, 1), "old", scheduled.append, lambda *_: None)
        assert backend.started.wait(1)
        getattr(worker, action)()
        backend.release.set()
        time.sleep(0.03)
        assert scheduled == []
        worker.close()


def test_transformers_backend_uses_official_prompt_shape(monkeypatch) -> None:
    captured: dict = {}

    class Tensor(np.ndarray):
        def cpu(self):
            return self

    def tensor(value):
        return np.asarray(value).view(Tensor)

    class Inputs(dict):
        def to(self, device):
            captured["device"] = device
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["processor_load"] = (model_id, kwargs)
            return cls()

        def __call__(self, **kwargs):
            captured["points"] = kwargs["input_points"]
            captured["labels"] = kwargs["input_labels"]
            return Inputs(original_sizes=tensor([[6, 8]]))

        def post_process_masks(self, masks, sizes):
            return [tensor([[[[False] * 8] * 6]])]

    class Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_load"] = (model_id, kwargs)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return types.SimpleNamespace(pred_masks=tensor([[[[0.0]]]]), iou_scores=tensor([[0.9]]))

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        inference_mode=lambda: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(Sam2Model=Model, Sam2Processor=Processor),
    )
    backend = TransformersSam2Backend("local-model", "auto", local_files_only=True)
    mask = backend.segment(np.zeros((6, 8, 3), np.uint8), (3, 2))
    assert mask.shape == (6, 8)
    assert captured["points"] == [[[[3, 2]]]]
    assert captured["labels"] == [[[1]]]
    assert captured["device"] == "cpu"
    assert captured["model_load"][1] == {"local_files_only": True}
