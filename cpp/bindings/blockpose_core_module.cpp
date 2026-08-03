// Optional nanobind adapter for the portable C++ scene-discovery core.
// This file intentionally knows only NumPy arrays and plain Python values;
// ROS, YAML, OpenCV, PCL, and model runtimes belong in separate adapters.
#include <blockpose/core/blockpose_core.hpp>
#ifdef BLOCKPOSE_CORE_WITH_VISION
#include <blockpose/vision/rgb_edge_prior.hpp>
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{
void validate_detection_parameters(const blockpose::core::DetectionParameters & params)
{
  const auto positive = [](double value, const char * name) {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
      }
    };
  positive(params.voxel_size, "voxel_size");
  positive(params.dbscan_eps, "dbscan_eps");
  positive(params.ground_thickness, "ground_thickness");
  positive(params.ground_ransac_distance, "ground_ransac_distance");
  if (!std::isfinite(params.ground_normal_min_z) || params.ground_normal_min_z < 0.0 ||
    params.ground_normal_min_z > 1.0)
  {
    throw std::invalid_argument("ground_normal_min_z must be finite and in [0, 1]");
  }
  positive(params.local_ground_cell_size, "local_ground_cell_size");
  if (!std::isfinite(params.local_ground_clearance) || params.local_ground_clearance < 0.0) {
    throw std::invalid_argument("local_ground_clearance must be finite and non-negative");
  }
  positive(params.cluster_min_extent_xy, "cluster_min_extent_xy");
  positive(params.cluster_max_extent_xy, "cluster_max_extent_xy");
  positive(params.region_max_extent_xy, "region_max_extent_xy");
  positive(params.cluster_min_extent_z, "cluster_min_extent_z");
  positive(params.cluster_max_extent_z, "cluster_max_extent_z");
  if (!std::isfinite(params.cluster_max_center_z) || params.cluster_max_center_z < 0.0) {
    throw std::invalid_argument("cluster_max_center_z must be finite and non-negative");
  }
  positive(params.ransac_distance, "ransac_distance");
  positive(params.refine_band, "refine_band");
  positive(params.refine_huber_scale, "refine_huber_scale");
  positive(params.refine_max_translation, "refine_max_translation");
  positive(params.refine_max_rotation_deg, "refine_max_rotation_deg");
  if (!std::isfinite(params.top_plane_angle_deg) || params.top_plane_angle_deg < 0.0 ||
    params.top_plane_angle_deg > 90.0 || !std::isfinite(params.side_plane_angle_deg) ||
    params.side_plane_angle_deg < 0.0 || params.side_plane_angle_deg > 90.0)
  {
    throw std::invalid_argument("plane angles must be finite and in [0, 90]");
  }
  if (!std::isfinite(params.min_score) || params.min_score < 0.0 ||
    !std::isfinite(params.fk_seed_min_evidence_gain) || params.fk_seed_min_evidence_gain < 0.0 ||
    params.ground_ransac_iterations == 0U ||
    params.ransac_iterations == 0U || params.ransac_search_max_points == 0U ||
    params.dbscan_min_points == 0U || params.cluster_min_size == 0U || params.min_inliers == 0U ||
    params.cluster_max_size < params.cluster_min_size ||
    params.cluster_max_extent_xy < params.cluster_min_extent_xy ||
    params.region_max_extent_xy < params.cluster_min_extent_xy ||
    params.cluster_max_extent_z < params.cluster_min_extent_z ||
    params.max_planes == 0U || params.refine_iterations == 0U || params.refine_min_points == 0U ||
    params.conflict_alternatives == 0U || params.proposal_max_components == 0U)
  {
    throw std::invalid_argument("DetectionParameters contains an invalid zero or non-finite threshold");
  }
  for (double dim : params.block_dims) {positive(dim, "block_dims");}
  for (const auto & dims : params.candidate_dims) {
    for (double dim : dims) {positive(dim, "candidate_dims");}
  }
}

template <typename Scalar>
blockpose::core::Points points_from_numpy(
  const nb::ndarray<nb::numpy, Scalar, nb::ndim<2>, nb::c_contig> & array,
  bool reject_nonfinite = true)
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
    if (reject_nonfinite && !point.allFinite()) {
      throw std::invalid_argument("points must be finite");
    }
    points.push_back(point);
  }
  return points;
}

blockpose::core::Point point_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & array, const char * name)
{
  if (array.shape(0) != 3) {throw std::invalid_argument(std::string(name) + " must have shape (3,)");}
  const blockpose::core::Point point(array(0), array(1), array(2));
  if (!point.allFinite()) {throw std::invalid_argument(std::string(name) + " must be finite");}
  return point;
}

Eigen::Matrix3d rotation_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & array)
{
  if (array.shape(0) != 3 || array.shape(1) != 3) {
    throw std::invalid_argument("rotation must have shape (3, 3)");
  }
  Eigen::Matrix3d rotation;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      rotation(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = array(row, column);
    }
  }
  if (!rotation.allFinite() ||
    !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1.0e-8) ||
    std::abs(rotation.determinant() - 1.0) > 1.0e-8)
  {
    throw std::invalid_argument("rotation must be a finite proper rotation matrix");
  }
  return rotation;
}

nb::dict refine_pose_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & points,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & position,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & rotation,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & dims,
  double huber_scale, double max_translation, double max_rotation_deg, std::size_t min_points)
{
  if (dims.shape(0) != 3) {throw std::invalid_argument("dims must have shape (3,)");}
  blockpose::core::CuboidPose initial;
  initial.position = point_from_numpy(position, "position");
  initial.rotation = rotation_from_numpy(rotation);
  for (std::size_t index = 0; index < 3; ++index) {
    if (!std::isfinite(dims(index)) || dims(index) <= 0.0) {
      throw std::invalid_argument("dims must be finite and positive");
    }
    initial.dims[index] = dims(index);
  }
  const blockpose::core::RefineParameters parameters{
    huber_scale, max_translation, max_rotation_deg, min_points, 200U};
  if (!std::isfinite(huber_scale) || huber_scale <= 0.0 || !std::isfinite(max_translation) ||
    max_translation <= 0.0 || !std::isfinite(max_rotation_deg) || max_rotation_deg <= 0.0 || min_points == 0U)
  {
    throw std::invalid_argument("refine parameters must be finite and positive");
  }
  const auto result = blockpose::core::refine_pose(initial, points_from_numpy(points), parameters);
  nb::dict output;
  nb::list refined_position, refined_rotation, correction;
  for (int row = 0; row < 3; ++row) {
    refined_position.append(result.pose.position[row]);
    nb::list rotation_row;
    for (int column = 0; column < 3; ++column) {rotation_row.append(result.pose.rotation(row, column));}
    refined_rotation.append(std::move(rotation_row));
  }
  for (double value : result.diagnostics.correction) {correction.append(value);}
  output["position"] = std::move(refined_position);
  output["rotation"] = std::move(refined_rotation);
  output["refined"] = result.refined;
  nb::dict diagnostics;
  diagnostics["attempted"] = result.diagnostics.attempted;
  diagnostics["point_count"] = result.diagnostics.point_count;
  diagnostics["evaluations"] = result.diagnostics.evaluations;
  diagnostics["iterations"] = result.diagnostics.iterations;
  diagnostics["initial_cost"] = result.diagnostics.initial_cost;
  diagnostics["final_cost"] = result.diagnostics.final_cost;
  diagnostics["correction"] = std::move(correction);
  diagnostics["translation_norm"] = result.diagnostics.translation_norm;
  diagnostics["rotation_deg"] = result.diagnostics.rotation_deg;
  diagnostics["guard_accepted"] = result.diagnostics.guard_accepted;
  output["diagnostics"] = std::move(diagnostics);
  return output;
}

nb::dict annotation_snap_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & points,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & seed,
  const blockpose::core::AnnotationSnapParameters & parameters)
{
  const auto result = blockpose::core::annotation_snap(
    points_from_numpy(points, false), point_from_numpy(seed, "seed"), parameters);
  nb::dict output; output["success"] = result.success; output["score"] = result.score;
  output["inlier_count"] = result.inlier_count; output["reason"] = result.reason;
  if (!result.success) {output["position"] = nb::none(); output["rotation"] = nb::none(); return output;}
  nb::list position, rotation;
  for (int row = 0; row < 3; ++row) {position.append(result.pose.position[row]); nb::list rotation_row; for (int column = 0; column < 3; ++column) rotation_row.append(result.pose.rotation(row, column)); rotation.append(std::move(rotation_row));}
  output["position"] = std::move(position); output["rotation"] = std::move(rotation); return output;
}

nb::object points_to_numpy(const blockpose::core::Points & points)
{
  auto numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("empty")(nb::make_tuple(points.size(), 3), numpy.attr("float64"));
  auto output = nb::cast<nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>>(
    array);
  for (std::size_t row = 0; row < points.size(); ++row) {
    for (std::size_t column = 0; column < 3; ++column) {output(row, column) = points[row](column);}
  }
  return array;
}

void validate_ground_parameters(const blockpose::core::GroundParameters & parameters)
{
  const auto positive = [](double value, const char * name) {
      if (!std::isfinite(value) || value <= 0.0) {throw std::invalid_argument(std::string(name) + " must be finite and positive");}
    };
  positive(parameters.thickness, "thickness"); positive(parameters.ransac_distance, "ransac_distance");
  positive(parameters.local_cell_size, "local_cell_size");
  if (!std::isfinite(parameters.z_percentile) || parameters.z_percentile < 0.0 || parameters.z_percentile > 100.0 ||
    !std::isfinite(parameters.normal_min_z) || parameters.normal_min_z < 0.0 || parameters.normal_min_z > 1.0 ||
    !std::isfinite(parameters.local_clearance) || parameters.local_clearance < 0.0 || parameters.ransac_iterations == 0U)
  {
    throw std::invalid_argument("GroundParameters contains an invalid range or zero iteration count");
  }
}

nb::dict remove_ground_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & input,
  const blockpose::core::GroundParameters & parameters)
{
  validate_ground_parameters(parameters);
  const auto points = points_from_numpy(input);
  blockpose::core::GroundRemovalResult result;
  { nb::gil_scoped_release release; result = blockpose::core::remove_ground(points, parameters); }
  nb::dict ground;
  ground["normal"] = nb::make_tuple(result.ground.normal.x(), result.ground.normal.y(), result.ground.normal.z());
  ground["offset"] = result.ground.offset;
  ground["source"] = result.ground.source;
  ground["cell_size"] = result.ground.cell_size;
  nb::dict output;
  output["points"] = points_to_numpy(result.above_ground);
  output["input_count"] = points.size();
  output["removed_count"] = points.size() - result.above_ground.size();
  output["ground"] = std::move(ground);
  return output;
}

nb::dict filter_workspace_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & input,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & minimum,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & maximum)
{
  const auto points = points_from_numpy(input); const auto low = point_from_numpy(minimum, "workspace minimum");
  const auto high = point_from_numpy(maximum, "workspace maximum");
  std::size_t removed = 0U;
  blockpose::core::Points kept;
  {nb::gil_scoped_release release; kept = blockpose::core::keep_points_inside_axis_aligned_box(points, low, high, &removed);}
  nb::dict output; output["points"] = points_to_numpy(kept); output["input_count"] = points.size(); output["removed_count"] = removed; return output;
}

nb::dict filter_exclusion_boxes_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & input,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & centers,
  const nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::c_contig> & rotations,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & sizes)
{
  if (centers.shape(1) != 3 || sizes.shape(1) != 3 || rotations.shape(1) != 3 || rotations.shape(2) != 3 || centers.shape(0) != sizes.shape(0) || centers.shape(0) != rotations.shape(0)) {throw std::invalid_argument("exclusion boxes require centers (N,3), rotations (N,3,3), and sizes (N,3)");}
  std::vector<blockpose::core::OrientedBox> boxes; boxes.reserve(centers.shape(0));
  for (std::size_t index = 0; index < centers.shape(0); ++index) {
    blockpose::core::OrientedBox box; for (std::size_t axis = 0; axis < 3; ++axis) {box.center(axis) = centers(index, axis); box.size(axis) = sizes(index, axis); for (std::size_t col = 0; col < 3; ++col) {box.rotation(axis, col) = rotations(index, axis, col);}} boxes.push_back(box);
  }
  const auto points = points_from_numpy(input); std::size_t removed = 0U; blockpose::core::Points kept;
  {nb::gil_scoped_release release; kept = blockpose::core::remove_points_inside_oriented_boxes(points, boxes, &removed);}
  nb::dict output; output["points"] = points_to_numpy(kept); output["input_count"] = points.size(); output["removed_count"] = removed; return output;
}

blockpose::core::SensorContext sensor_context_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & origin,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & ray_directions,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & ranges,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & miss_directions,
  const nb::object & ray_origins, const nb::object & miss_origins)
{
  blockpose::core::SensorContext context;
  context.origin = point_from_numpy(origin, "origin");
  context.ray_directions = points_from_numpy(ray_directions);
  if (ranges.shape(0) != ray_directions.shape(0)) {throw std::invalid_argument("ranges must match ray_directions");}
  context.ranges.reserve(ranges.shape(0));
  for (std::size_t i = 0; i < ranges.shape(0); ++i) {context.ranges.push_back(ranges(i));}
  context.miss_directions = points_from_numpy(miss_directions);
  if (!ray_origins.is_none()) {context.ray_origins = points_from_numpy(nb::cast<nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>>(ray_origins));}
  if (!miss_origins.is_none()) {context.miss_origins = points_from_numpy(nb::cast<nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>>(miss_origins));}
  return blockpose::core::validate_sensor_context(std::move(context));
}

#ifdef BLOCKPOSE_CORE_WITH_VISION
void validate_rgb_edge_prior_parameters(const blockpose::vision::RgbEdgePriorParameters & params)
{
  const auto positive = [](double value, const char * name) {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
      }
    };
  positive(params.weight, "weight");
  if (!std::isfinite(params.edge_percentile) || params.edge_percentile <= 0.0 ||
    params.edge_percentile >= 100.0)
  {
    throw std::invalid_argument("edge_percentile must be finite and in (0, 100)");
  }
  positive(params.sample_spacing_px, "sample_spacing_px");
  positive(params.distance_scale_px, "distance_scale_px");
  if (!std::isfinite(params.min_support_fraction) || params.min_support_fraction < 0.0 ||
    params.min_support_fraction > 1.0 || params.min_samples < 1 ||
    params.max_image_dimension_px < 2)
  {
    throw std::invalid_argument("RgbEdgePriorParameters contains an invalid threshold");
  }
}

cv::Matx33d camera_matrix_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & array)
{
  if (array.shape(0) != 3 || array.shape(1) != 3) {
    throw std::invalid_argument("camera_matrix must have shape (3, 3)");
  }
  cv::Matx33d result;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      if (!std::isfinite(array(row, column))) {
        throw std::invalid_argument("camera_matrix must be finite");
      }
      result(static_cast<int>(row), static_cast<int>(column)) = array(row, column);
    }
  }
  if (result(0, 0) <= 0.0 || result(1, 1) <= 0.0) {
    throw std::invalid_argument("camera_matrix focal lengths must be positive");
  }
  return result;
}

Eigen::Isometry3d world_from_camera_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & array)
{
  if (array.shape(0) != 4 || array.shape(1) != 4) {
    throw std::invalid_argument("world_from_camera must have shape (4, 4)");
  }
  Eigen::Matrix4d matrix;
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      if (!std::isfinite(array(row, column))) {
        throw std::invalid_argument("world_from_camera must be finite");
      }
      matrix(static_cast<int>(row), static_cast<int>(column)) = array(row, column);
    }
  }
  const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
  if (!matrix.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-12) ||
    !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
    rotation.determinant() <= 0.0)
  {
    throw std::invalid_argument("world_from_camera must be a rigid homogeneous transform");
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.matrix() = matrix;
  return result;
}

blockpose::core::Pose pose_from_numpy(
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & position,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & rotation,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & dims)
{
  blockpose::core::Pose pose;
  pose.position = point_from_numpy(position, "position");
  if (rotation.shape(0) != 3 || rotation.shape(1) != 3 || dims.shape(0) != 3) {
    throw std::invalid_argument("rotation must have shape (3, 3) and dims must have shape (3,)");
  }
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      if (!std::isfinite(rotation(row, column))) {throw std::invalid_argument("rotation must be finite");}
      pose.rotation(static_cast<int>(row), static_cast<int>(column)) = rotation(row, column);
    }
    if (!std::isfinite(dims(row)) || dims(row) <= 0.0) {
      throw std::invalid_argument("dims must be finite and positive");
    }
    pose.dims[row] = dims(row);
  }
  if (!(pose.rotation.transpose() * pose.rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-8) ||
    pose.rotation.determinant() <= 0.0)
  {
    throw std::invalid_argument("rotation must be a proper rotation matrix");
  }
  return pose;
}

blockpose::vision::RgbEdgePrior make_rgb_edge_prior(
  const nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>, nb::c_contig> & grayscale,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & camera_matrix,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & distortion,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & world_from_camera,
  const nb::object & excluded_mask, const blockpose::vision::RgbEdgePriorParameters & parameters)
{
  if (grayscale.shape(0) < 2 || grayscale.shape(1) < 2) {
    throw std::invalid_argument("grayscale must have shape (H, W) with H and W at least 2");
  }
  if (distortion.shape(0) != 0 && distortion.shape(0) != 4 && distortion.shape(0) != 5 && distortion.shape(0) != 8 &&
    distortion.shape(0) != 12 && distortion.shape(0) != 14)
  {
    throw std::invalid_argument("distortion must contain 0, 4, 5, 8, 12, or 14 coefficients");
  }
  cv::Mat distortion_mat;
  if (distortion.shape(0) != 0) {
    distortion_mat = cv::Mat(1, static_cast<int>(distortion.shape(0)), CV_64F);
  }
  for (std::size_t index = 0; index < distortion.shape(0); ++index) {
    if (!std::isfinite(distortion(index))) {throw std::invalid_argument("distortion must be finite");}
    distortion_mat.at<double>(0, static_cast<int>(index)) = distortion(index);
  }
  cv::Mat mask;
  if (!excluded_mask.is_none()) {
    const auto input = nb::cast<nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>, nb::c_contig>>(excluded_mask);
    if (input.shape(0) != grayscale.shape(0) || input.shape(1) != grayscale.shape(1)) {
      throw std::invalid_argument("excluded_mask must match grayscale shape");
    }
    mask = cv::Mat(static_cast<int>(input.shape(0)), static_cast<int>(input.shape(1)), CV_8UC1,
      const_cast<std::uint8_t *>(input.data())).clone();
  }
  validate_rgb_edge_prior_parameters(parameters);
  const cv::Mat image(static_cast<int>(grayscale.shape(0)), static_cast<int>(grayscale.shape(1)), CV_8UC1,
    const_cast<std::uint8_t *>(grayscale.data()));
  return {image, camera_matrix_from_numpy(camera_matrix), distortion_mat,
    world_from_camera_from_numpy(world_from_camera), std::move(mask), parameters};
}

nb::dict score_rgb_edge_prior(
  const blockpose::vision::RgbEdgePrior & prior,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & position,
  const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & rotation,
  const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & dims)
{
  const auto evidence = prior.score(pose_from_numpy(position, rotation, dims));
  nb::dict result;
  result["score"] = evidence.score;
  result["available"] = evidence.available;
  return result;
}
#endif

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

nb::tuple point_to_tuple(const blockpose::core::Point & point)
{
  return nb::make_tuple(point.x(), point.y(), point.z());
}

nb::dict evidence_to_dict(const blockpose::core::HypothesisEvidence & evidence)
{
  nb::dict result;
  result["support_points"] = evidence.support_points;
  result["top_height_error_m"] = evidence.top_height_error_m;
  result["score"] = evidence.score;
  result["expected_visible_faces"] = evidence.expected_visible_faces;
  result["covered_visible_faces"] = evidence.covered_visible_faces;
  result["free_space_violations"] = evidence.free_space_violations;
  result["supported_rays"] = evidence.supported_rays;
  result["observed_geometry_faces"] = evidence.observed_geometry_faces;
  result["incident_rays"] = evidence.incident_rays;
  return result;
}

nb::dict lineage_to_dict(const blockpose::core::RawHypothesisLineage & lineage)
{
  nb::dict result;
  result["id"] = lineage.id;
  result["proposal_component"] = lineage.proposal_component;
  result["region"] = lineage.region;
  result["top_plane"] = lineage.top_plane;
  result["side_plane"] = lineage.side_plane ? nb::cast(*lineage.side_plane) : nb::none();
  result["top_normal"] = point_to_tuple(lineage.top_normal);
  result["top_centroid"] = point_to_tuple(lineage.top_centroid);
  result["side_normal"] = lineage.side_normal ? nb::cast(point_to_tuple(*lineage.side_normal)) : nb::none();
  result["side_centroid"] = lineage.side_centroid ? nb::cast(point_to_tuple(*lineage.side_centroid)) : nb::none();
  result["candidate_dims"] = lineage.candidate_dims;
  result["top_only"] = lineage.top_only;
  result["synthesized_pose"] = pose_to_dict(lineage.synthesized_pose);
  result["evidence"] = evidence_to_dict(lineage.evidence);
  result["top_support_height_m"] = lineage.top_support_height_m;
  result["accepted_to_raw"] = lineage.accepted_to_raw;
  result["passed_score_threshold"] = lineage.passed_score_threshold;
  result["fate"] = lineage.fate;
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

  nb::dict timings;
  timings["voxel_downsample"] = detection.timings.voxel_downsample;
  timings["ground_model"] = detection.timings.ground_model;
  timings["proposals"] = detection.timings.proposals;
  timings["region_split"] = detection.timings.region_split;
  timings["plane_search"] = detection.timings.plane_search;
  timings["plane_full_score_refit"] = detection.timings.plane_full_score_refit;
  timings["hypothesis_generation_evidence"] = detection.timings.hypothesis_generation_evidence;
  timings["plane_hypotheses"] = detection.timings.plane_hypotheses;
  timings["selection"] = detection.timings.selection;
  timings["refinement"] = detection.timings.refinement;
  timings["total"] = detection.timings.total;
  result["timings_ms"] = std::move(timings);

  nb::list diagnostics;
  for (const auto & diagnostic : detection.diagnostics) {
    nb::dict entry;
    entry["evidence"] = evidence_to_dict(diagnostic.evidence);
    nb::dict features;
    features["support"] = diagnostic.features[0];
    features["height_consistency"] = diagnostic.features[1];
    features["visible_face_coverage"] = diagnostic.features[2];
    features["free_space_consistency"] = diagnostic.features[3];
    features["surface_explained"] = diagnostic.features[4];
    entry["features"] = std::move(features);
    entry["observability"] = diagnostic.observability;
    // Native calibrated confidence has no artifact-backed implementation yet.
    entry["calibrated_confidence"] = nb::none();
    entry["proposal_scale_m"] = diagnostic.proposal_scale_m ?
      nb::cast(*diagnostic.proposal_scale_m) : nb::none();
    entry["support_height_m"] = diagnostic.support_height_m;
    diagnostics.append(std::move(entry));
  }
  result["diagnostics"] = std::move(diagnostics);

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
  nb::list raw_lineage;
  for (const auto & entry : detection.raw_lineage) {
    raw_lineage.append(lineage_to_dict(entry));
  }
  result["raw_lineage"] = std::move(raw_lineage);
  return result;
}

template <typename Scalar>
nb::dict discover(
  const nb::ndarray<nb::numpy, Scalar, nb::ndim<2>, nb::c_contig> & input,
  const blockpose::core::DetectionParameters & params)
{
  // Copy before releasing the GIL: NumPy owns the input buffer and another
  // Python thread may otherwise resize or replace it while detection runs.
  validate_detection_parameters(params);
  const blockpose::core::Points points = points_from_numpy(input);
  blockpose::core::DetectionResult detection;
  {
    nb::gil_scoped_release release;
    detection = blockpose::core::detect(points, params);
  }
  return result_to_dict(detection);
}

template <typename Scalar>
nb::dict discover_with_context(
  const nb::ndarray<nb::numpy, Scalar, nb::ndim<2>, nb::c_contig> & input,
  const blockpose::core::DetectionParameters & params, const nb::object & context)
{
  validate_detection_parameters(params);
  const blockpose::core::Points points = points_from_numpy(input);
  blockpose::core::DetectionResult detection;
  if (context.is_none()) {
    nb::gil_scoped_release release;
    detection = blockpose::core::detect(points, params);
  } else {
    const auto sensor_context = nb::cast<blockpose::core::SensorContext>(context);
    nb::gil_scoped_release release;
    detection = blockpose::core::detect(points, params, &sensor_context);
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
  auto parameters = nb::class_<blockpose::core::DetectionParameters>(module, "DetectionParameters");
  parameters.def(nb::init<>())
    .def_rw("voxel_size", &blockpose::core::DetectionParameters::voxel_size)
    .def_rw("dbscan_eps", &blockpose::core::DetectionParameters::dbscan_eps)
    .def_rw("ground_thickness", &blockpose::core::DetectionParameters::ground_thickness)
    .def_rw("ground_ransac_distance", &blockpose::core::DetectionParameters::ground_ransac_distance)
    .def_rw("ground_ransac_iterations", &blockpose::core::DetectionParameters::ground_ransac_iterations)
    .def_rw("ground_normal_min_z", &blockpose::core::DetectionParameters::ground_normal_min_z)
    .def_rw("local_ground_cell_size", &blockpose::core::DetectionParameters::local_ground_cell_size)
    .def_rw("local_ground_clearance", &blockpose::core::DetectionParameters::local_ground_clearance)
    .def_rw("dbscan_min_points", &blockpose::core::DetectionParameters::dbscan_min_points)
    .def_rw("cluster_min_size", &blockpose::core::DetectionParameters::cluster_min_size)
    .def_rw("cluster_max_size", &blockpose::core::DetectionParameters::cluster_max_size)
    .def_rw("cluster_min_extent_xy", &blockpose::core::DetectionParameters::cluster_min_extent_xy)
    .def_rw("cluster_max_extent_xy", &blockpose::core::DetectionParameters::cluster_max_extent_xy)
    .def_rw("region_max_extent_xy", &blockpose::core::DetectionParameters::region_max_extent_xy)
    .def_rw("cluster_min_extent_z", &blockpose::core::DetectionParameters::cluster_min_extent_z)
    .def_rw("cluster_max_extent_z", &blockpose::core::DetectionParameters::cluster_max_extent_z)
    .def_rw("cluster_max_center_z", &blockpose::core::DetectionParameters::cluster_max_center_z)
    .def_rw("ransac_distance", &blockpose::core::DetectionParameters::ransac_distance)
    .def_rw("ransac_iterations", &blockpose::core::DetectionParameters::ransac_iterations)
    .def_rw("ransac_search_max_points", &blockpose::core::DetectionParameters::ransac_search_max_points)
    .def_rw("ransac_seed", &blockpose::core::DetectionParameters::ransac_seed)
    .def_rw("max_planes", &blockpose::core::DetectionParameters::max_planes)
    .def_rw("min_inliers", &blockpose::core::DetectionParameters::min_inliers)
    .def_rw("top_plane_angle_deg", &blockpose::core::DetectionParameters::top_plane_angle_deg)
    .def_rw("side_plane_angle_deg", &blockpose::core::DetectionParameters::side_plane_angle_deg)
    .def_rw("block_dims", &blockpose::core::DetectionParameters::block_dims)
    .def_rw("candidate_dims", &blockpose::core::DetectionParameters::candidate_dims)
    .def_rw("refine_band", &blockpose::core::DetectionParameters::refine_band)
    .def_rw("refine_iterations", &blockpose::core::DetectionParameters::refine_iterations)
    .def_rw("refine_min_points", &blockpose::core::DetectionParameters::refine_min_points)
    .def_rw("refine_huber_scale", &blockpose::core::DetectionParameters::refine_huber_scale)
    .def_rw("refine_max_translation", &blockpose::core::DetectionParameters::refine_max_translation)
    .def_rw("refine_max_rotation_deg", &blockpose::core::DetectionParameters::refine_max_rotation_deg)
    .def_rw("refine_preserve_top_axis_if_gravity_worsens", &blockpose::core::DetectionParameters::refine_preserve_top_axis_if_gravity_worsens)
    .def_rw("min_score", &blockpose::core::DetectionParameters::min_score)
    .def_rw("fk_seed_min_evidence_gain", &blockpose::core::DetectionParameters::fk_seed_min_evidence_gain)
    .def_rw("fit_wide_proposals", &blockpose::core::DetectionParameters::fit_wide_proposals)
    .def_rw("conflict_alternatives", &blockpose::core::DetectionParameters::conflict_alternatives)
    .def_rw("proposal_max_components", &blockpose::core::DetectionParameters::proposal_max_components);
  nb::class_<blockpose::core::SensorContext>(module, "SensorContext")
    .def_static("from_arrays", &sensor_context_from_numpy, nb::arg("origin"),
      nb::arg("ray_directions"), nb::arg("ranges"), nb::arg("miss_directions"),
      nb::arg("ray_origins") = nb::none(), nb::arg("miss_origins") = nb::none(),
      "Build and validate ray/free-space evidence from contiguous float64 NumPy arrays.");
  module.def("discover", &discover<float>, nb::arg("points"), nb::arg("params") = blockpose::core::DetectionParameters{},
    "Run geometry-only scene discovery using typed C++ DetectionParameters.");
  module.def("discover", &discover<double>, nb::arg("points"), nb::arg("params") = blockpose::core::DetectionParameters{},
    "Run geometry-only scene discovery using typed C++ DetectionParameters.");
  module.def("discover", &discover_with_context<float>, nb::arg("points"), nb::arg("params"), nb::arg("sensor_context"),
    "Run discovery with typed parameters and optional validated SensorContext.");
  module.def("discover", &discover_with_context<double>, nb::arg("points"), nb::arg("params"), nb::arg("sensor_context"),
    "Run discovery with typed parameters and optional validated SensorContext.");
  module.def("refine_pose", &refine_pose_from_numpy, nb::arg("points"), nb::arg("position"),
    nb::arg("rotation"), nb::arg("dims"), nb::arg("huber_scale") = 0.05,
    nb::arg("max_translation") = 0.15, nb::arg("max_rotation_deg") = 20.0,
    nb::arg("min_points") = 20U,
    "Refine one cuboid pose with the portable C++ solver and return pre-guard diagnostics.");
  auto snap_parameters = nb::class_<blockpose::core::AnnotationSnapParameters>(module, "AnnotationSnapParameters");
  snap_parameters.def(nb::init<>())
    .def_rw("crop_radius_m", &blockpose::core::AnnotationSnapParameters::crop_radius_m)
    .def_rw("voxel_m", &blockpose::core::AnnotationSnapParameters::voxel_m)
    .def_rw("plane_tolerance_m", &blockpose::core::AnnotationSnapParameters::plane_tolerance_m)
    .def_rw("surface_tolerance_m", &blockpose::core::AnnotationSnapParameters::surface_tolerance_m)
    .def_rw("min_points", &blockpose::core::AnnotationSnapParameters::min_points)
    .def_rw("min_inliers", &blockpose::core::AnnotationSnapParameters::min_inliers)
    .def_rw("min_score", &blockpose::core::AnnotationSnapParameters::min_score)
    .def_rw("ambiguity_score_ratio", &blockpose::core::AnnotationSnapParameters::ambiguity_score_ratio)
    .def_rw("ambiguity_distance_m", &blockpose::core::AnnotationSnapParameters::ambiguity_distance_m)
    .def_rw("reject_ambiguous", &blockpose::core::AnnotationSnapParameters::reject_ambiguous);
  module.def("annotation_snap", &annotation_snap_from_numpy, nb::arg("points"), nb::arg("seed"),
    nb::arg("parameters") = blockpose::core::AnnotationSnapParameters{}, "Fit a canonical block pose around an annotation click.");
  auto ground_parameters = nb::class_<blockpose::core::GroundParameters>(module, "GroundParameters");
  ground_parameters.def(nb::init<>())
    .def_rw("z_percentile", &blockpose::core::GroundParameters::z_percentile)
    .def_rw("thickness", &blockpose::core::GroundParameters::thickness)
    .def_rw("ransac_distance", &blockpose::core::GroundParameters::ransac_distance)
    .def_rw("normal_min_z", &blockpose::core::GroundParameters::normal_min_z)
    .def_rw("ransac_iterations", &blockpose::core::GroundParameters::ransac_iterations)
    .def_rw("ransac_seed", &blockpose::core::GroundParameters::ransac_seed)
    .def_rw("local_cell_size", &blockpose::core::GroundParameters::local_cell_size)
    .def_rw("local_clearance", &blockpose::core::GroundParameters::local_clearance);
  module.def("remove_ground", &remove_ground_from_numpy, nb::arg("points"),
    nb::arg("parameters") = blockpose::core::GroundParameters{},
    "Remove C++-modelled ground and return retained points plus model diagnostics.");
  module.def("filter_workspace", &filter_workspace_from_numpy, nb::arg("points"), nb::arg("minimum"), nb::arg("maximum"),
    "Keep points inside an inclusive axis-aligned workspace ROI.");
  module.def("filter_exclusion_boxes", &filter_exclusion_boxes_from_numpy, nb::arg("points"), nb::arg("centers"), nb::arg("rotations"), nb::arg("sizes"),
    "Remove points inside explicit oriented exclusion boxes.");
#ifdef BLOCKPOSE_CORE_WITH_VISION
  auto rgb_parameters = nb::class_<blockpose::vision::RgbEdgePriorParameters>(
    module, "RgbEdgePriorParameters");
  rgb_parameters.def(nb::init<>())
    .def_rw("weight", &blockpose::vision::RgbEdgePriorParameters::weight)
    .def_rw("edge_percentile", &blockpose::vision::RgbEdgePriorParameters::edge_percentile)
    .def_rw("sample_spacing_px", &blockpose::vision::RgbEdgePriorParameters::sample_spacing_px)
    .def_rw("distance_scale_px", &blockpose::vision::RgbEdgePriorParameters::distance_scale_px)
    .def_rw("min_support_fraction", &blockpose::vision::RgbEdgePriorParameters::min_support_fraction)
    .def_rw("min_samples", &blockpose::vision::RgbEdgePriorParameters::min_samples)
    .def_rw("max_image_dimension_px", &blockpose::vision::RgbEdgePriorParameters::max_image_dimension_px);
  nb::class_<blockpose::vision::RgbEdgePrior>(module, "RgbEdgePrior")
    .def(nb::new_([](
      const nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>, nb::c_contig> & grayscale,
      const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & camera_matrix,
      const nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> & distortion,
      const nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig> & world_from_camera,
      const nb::object & excluded_mask, const blockpose::vision::RgbEdgePriorParameters & parameters) {
        return new blockpose::vision::RgbEdgePrior(make_rgb_edge_prior(
            grayscale, camera_matrix, distortion, world_from_camera, excluded_mask, parameters));
      }), nb::arg("grayscale"), nb::arg("camera_matrix"),
      nb::arg("distortion"), nb::arg("world_from_camera"), nb::arg("excluded_mask") = nb::none(),
      nb::arg("parameters") = blockpose::vision::RgbEdgePriorParameters{},
      "Create a reusable native scorer from uint8 grayscale image and calibrated camera data.")
    .def("ready", &blockpose::vision::RgbEdgePrior::ready)
    .def("score", &score_rgb_edge_prior, nb::arg("position"), nb::arg("rotation"), nb::arg("dims"),
      "Return native visual evidence for one finite, rigid cuboid pose.");
  module.attr("vision_enabled") = true;
#else
  module.attr("vision_enabled") = false;
#endif
}
