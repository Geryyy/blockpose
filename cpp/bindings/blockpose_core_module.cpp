// Optional nanobind adapter for the portable C++ scene-discovery core.
// This file intentionally knows only NumPy arrays and plain Python values;
// ROS, YAML, OpenCV, PCL, and model runtimes belong in separate adapters.
#include <blockpose/core/blockpose_core.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{
template <typename Scalar>
blockpose::core::Points points_from_numpy(
  const nb::ndarray<nb::numpy, Scalar, nb::ndim<2>, nb::c_contig> & array)
{
  if (array.shape(1) != 3) {
    throw std::invalid_argument("points must have shape (N, 3)");
  }

  blockpose::core::Points points;
  points.reserve(array.shape(0));
  for (std::size_t row = 0; row < array.shape(0); ++row) {
    const blockpose::core::Point point(
      static_cast<double>(array(row, 0)),
      static_cast<double>(array(row, 1)),
      static_cast<double>(array(row, 2)));
    if (!point.allFinite()) {
      throw std::invalid_argument("points must be finite");
    }
    points.push_back(point);
  }
  return points;
}

nb::dict pose_to_dict(const blockpose::core::Pose & pose)
{
  nb::dict result;
  result["position"] = nb::make_tuple(pose.position.x(), pose.position.y(), pose.position.z());
  nb::list rotation;
  for (int row = 0; row < 3; ++row) {
    rotation.append(nb::make_tuple(
      pose.rotation(row, 0), pose.rotation(row, 1), pose.rotation(row, 2)));
  }
  result["rotation"] = std::move(rotation);
  result["dims"] = pose.dims;
  result["confidence"] = pose.confidence;
  return result;
}

nb::dict result_to_dict(const blockpose::core::DetectionResult & detection)
{
  nb::dict result;
  nb::list poses;
  for (const auto & pose : detection.poses) {
    poses.append(pose_to_dict(pose));
  }
  result["poses"] = std::move(poses);
  result["count"] = detection.poses.size();

  nb::dict counts;
  counts["input_points"] = detection.counts.input_points;
  counts["downsampled_points"] = detection.counts.downsampled_points;
  counts["above_support_points"] = detection.counts.above_support_points;
  counts["proposal_components"] = detection.counts.proposal_components;
  counts["plane_regions"] = detection.counts.plane_regions;
  counts["plane_fit_calls"] = detection.counts.plane_fit_calls;
  counts["plane_search_points"] = detection.counts.plane_search_points;
  counts["plane_full_points_scored"] = detection.counts.plane_full_points_scored;
  counts["plane_trials_evaluated"] = detection.counts.plane_trials_evaluated;
  counts["plane_valid_trials"] = detection.counts.plane_valid_trials;
  counts["raw_hypotheses"] = detection.counts.raw_hypotheses;
  counts["refinement_candidates"] = detection.counts.refinement_candidates;
  counts["selected_hypotheses"] = detection.counts.selected_hypotheses;
  result["counts"] = std::move(counts);

  nb::list refined_candidate_trace;
  for (const auto & entry : detection.refined_candidate_trace) {
    nb::dict trace;
    trace["id"] = entry.id;
    trace["source"] = entry.source;
    trace["lineage_id"] = entry.lineage_id;
    trace["pose"] = pose_to_dict(entry.pose);
    trace["score"] = entry.evidence.score;
    trace["support_points"] = entry.evidence.support_points;
    trace["selection_score"] = entry.selection_score;
    trace["stage"] = entry.stage;
    trace["fate"] = entry.fate;
    refined_candidate_trace.append(std::move(trace));
  }
  result["refined_candidate_trace"] = std::move(refined_candidate_trace);
  return result;
}

template <typename Scalar>
nb::dict discover(
  const nb::ndarray<nb::numpy, Scalar, nb::ndim<2>, nb::c_contig> & input)
{
  // Copy before releasing the GIL: NumPy owns the input buffer and another
  // Python thread may otherwise resize or replace it while detection runs.
  const blockpose::core::Points points = points_from_numpy(input);
  blockpose::core::DetectionResult detection;
  {
    nb::gil_scoped_release release;
    detection = blockpose::core::detect(points);
  }
  return result_to_dict(detection);
}
}  // namespace

NB_MODULE(_core, module)
{
  module.doc() = "Optional NumPy bridge for the portable blockpose C++ core.";
  module.attr("__version__") = BLOCKPOSE_CORE_VERSION;
  module.def("version", []() { return std::string(BLOCKPOSE_CORE_VERSION); },
    "Return the C++ blockpose_core build version.");
  module.def("discover", &discover<float>, nb::arg("points"),
    "Run default geometry-only scene discovery on a contiguous float32 (N, 3) NumPy array.");
  module.def("discover", &discover<double>, nb::arg("points"),
    "Run default geometry-only scene discovery on a contiguous float64 (N, 3) NumPy array.");
}
