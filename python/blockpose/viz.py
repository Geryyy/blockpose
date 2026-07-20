"""Interactive 3D visualization of detections against ground truth.

Plotly rather than matplotlib: the point of this view is to rotate and zoom
around a partially observed scene to see *why* a detection is where it is, which
a static projection cannot show. Figures are self-contained HTML, so they render
in JupyterLab, VS Code, and nbviewer without a widget extension.

Ground truth is green, detections are blue, and the sensor is marked with the
rays implied by its origin. Where a detection sits exactly on its ground-truth
box the two wireframes coincide -- that is the expected appearance after model
refinement, and any visible gap is a real error, not a rendering offset.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from blockpose.pose import BlockPose
from blockpose.synth import BLOCK_DIMS

GT_COLOR = "#00c853"
DET_COLOR = "#2979ff"
SENSOR_COLOR = "#ff6d00"
# 12 edges of a cuboid as vertex-index pairs, in the corner order built below.
_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def box_corners(pose: BlockPose, dims=BLOCK_DIMS) -> np.ndarray:
    """Eight world-frame corners of the posed cuboid, (8, 3)."""
    half = np.asarray(dims, dtype=np.float64) / 2.0
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
    )
    return (signs * half) @ pose.rotation.T + pose.position


def _wireframe(poses, dims, color, name, width=4):
    """One trace holding every box's edges, separated by NaN breaks."""
    xs, ys, zs = [], [], []
    for pose in poses:
        corners = box_corners(pose, dims)
        for start, end in _EDGES:
            xs += [corners[start, 0], corners[end, 0], None]
            ys += [corners[start, 1], corners[end, 1], None]
            zs += [corners[start, 2], corners[end, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        hoverinfo="name",
    )


def _equal_aspect(points: np.ndarray) -> dict:
    """Cubic axis ranges so blocks are not visually distorted."""
    lower, upper = points.min(axis=0), points.max(axis=0)
    centre = (lower + upper) / 2.0
    reach = float(np.max(upper - lower)) / 2.0 or 1.0
    axis = lambda index: {  # noqa: E731
        "range": [centre[index] - reach, centre[index] + reach],
        "showbackground": False,
    }
    return {
        "xaxis": {**axis(0), "title": "x [m]"},
        "yaxis": {**axis(1), "title": "y [m]"},
        "zaxis": {**axis(2), "title": "z [m]"},
        "aspectmode": "cube",
    }


def scene_figure(
    points: np.ndarray,
    ground_truth: list[BlockPose] | None = None,
    detections: list[BlockPose] | None = None,
    *,
    dims=BLOCK_DIMS,
    sensor_origin: np.ndarray | None = None,
    title: str = "",
    max_points: int = 20000,
    point_size: float = 1.2,
    height: int = 700,
) -> go.Figure:
    """Interactive cloud + ground-truth + detection view.

    The cloud is subsampled to `max_points` (deterministically) because a browser
    renders roughly 20k 3D markers smoothly and a raycast scene can hold ten
    times that. Subsampling is uniform, so coverage is preserved.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    shown = points
    if len(points) > max_points:
        stride = int(np.ceil(len(points) / max_points))
        shown = points[::stride]

    traces = [
        go.Scatter3d(
            x=shown[:, 0], y=shown[:, 1], z=shown[:, 2],
            mode="markers",
            marker={
                "size": point_size,
                "color": shown[:, 2],
                "colorscale": "Viridis",
                "opacity": 0.85,
                "showscale": False,
            },
            name=f"cloud ({len(shown):,} of {len(points):,})",
            hoverinfo="skip",
        )
    ]
    if ground_truth:
        traces.append(
            _wireframe(ground_truth, dims, GT_COLOR, f"ground truth ({len(ground_truth)})")
        )
    if detections:
        traces.append(_wireframe(detections, dims, DET_COLOR, f"detections ({len(detections)})"))
    if sensor_origin is not None:
        origin = np.asarray(sensor_origin, dtype=np.float64)
        traces.append(
            go.Scatter3d(
                x=[origin[0]], y=[origin[1]], z=[origin[2]],
                mode="markers",
                marker={"size": 6, "color": SENSOR_COLOR, "symbol": "diamond"},
                name="sensor",
            )
        )

    extent = np.vstack([points] + [box_corners(p, dims) for p in (ground_truth or [])])
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=title,
        height=height,
        margin={"l": 0, "r": 0, "t": 40 if title else 10, "b": 0},
        scene=_equal_aspect(extent),
        legend={"yanchor": "top", "y": 0.98, "xanchor": "left", "x": 0.01},
        showlegend=True,
    )
    return figure


def sample_figure(sample, detections=None, *, dims=BLOCK_DIMS, title="", **kwargs) -> go.Figure:
    """`scene_figure` for a `SceneSample`, wiring up ground truth and sensor."""
    return scene_figure(
        sample.points,
        sample.ground_truth,
        detections,
        dims=dims,
        sensor_origin=getattr(sample, "sensor_origin", None),
        title=title,
        **kwargs,
    )


def error_table(detections, ground_truth, max_match_dist: float = 0.5) -> list[dict]:
    """Per-match translation and symmetry-aware rotation error, plus misses.

    Kept next to the figure so a notebook can show the numbers that explain what
    the 3D view is displaying, rather than eyeballing overlap.
    """
    from blockpose.evaluate import match_detections, rotation_error_deg, translation_error

    pairs = match_detections(detections, ground_truth, max_dist=max_match_dist)
    matched_gt = {gt_index for _, gt_index in pairs}
    rows = [
        {
            "gt": gt_index,
            "detection": det_index,
            "translation_mm": 1000.0 * translation_error(detections[det_index],
                                                         ground_truth[gt_index]),
            "rotation_deg": rotation_error_deg(detections[det_index], ground_truth[gt_index]),
        }
        for det_index, gt_index in sorted(pairs, key=lambda pair: pair[1])
    ]
    rows += [
        {"gt": index, "detection": None, "translation_mm": None, "rotation_deg": None}
        for index in range(len(ground_truth))
        if index not in matched_gt
    ]
    return rows
