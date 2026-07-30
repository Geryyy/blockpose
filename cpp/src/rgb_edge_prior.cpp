#include <blockpose/vision/rgb_edge_prior.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace blockpose::vision
{
namespace
{
std::array<std::array<int, 3>, 8> corner_signs()
{
  return {{{{-1, -1, -1}}, {{-1, -1, 1}}, {{-1, 1, -1}}, {{-1, 1, 1}},
    {{1, -1, -1}}, {{1, -1, 1}}, {{1, 1, -1}}, {{1, 1, 1}}}};
}

std::array<Eigen::Vector3d, 8> corners_world(const core::Pose & pose)
{
  std::array<Eigen::Vector3d, 8> result;
  const auto signs = corner_signs();
  for (std::size_t index = 0; index < result.size(); ++index) {
    const Eigen::Vector3d local(
      0.5 * signs[index][0] * pose.dims[0], 0.5 * signs[index][1] * pose.dims[1],
      0.5 * signs[index][2] * pose.dims[2]);
    result[index] = pose.rotation * local + pose.position;
  }
  return result;
}
}  // namespace

RgbEdgePrior::RgbEdgePrior(
  const cv::Mat & grayscale, const cv::Matx33d & camera_matrix,
  const cv::Mat & distortion, const Eigen::Isometry3d & world_from_camera,
  cv::Mat excluded_mask, RgbEdgePriorParameters parameters)
: camera_matrix_(camera_matrix), distortion_(distortion.clone()),
  world_from_camera_(world_from_camera), excluded_mask_(std::move(excluded_mask)),
  parameters_(parameters), source_size_(grayscale.cols, grayscale.rows)
{
  if (grayscale.empty() || grayscale.type() != CV_8UC1 || grayscale.cols < 2 || grayscale.rows < 2 ||
    !std::isfinite(parameters_.weight) || parameters_.weight <= 0.0 ||
    parameters_.edge_percentile <= 0.0 || parameters_.edge_percentile >= 100.0 ||
    parameters_.sample_spacing_px <= 0.0 || parameters_.distance_scale_px <= 0.0 ||
    parameters_.min_samples < 1 || parameters_.max_image_dimension_px < 2 ||
    parameters_.min_support_fraction < 0.0 || parameters_.min_support_fraction > 1.0) {return;}

  cv::Mat working;
  const int largest = std::max(grayscale.cols, grayscale.rows);
  if (largest > parameters_.max_image_dimension_px) {
    const double scale = static_cast<double>(parameters_.max_image_dimension_px) / largest;
    cv::resize(grayscale, working, cv::Size(), scale, scale, cv::INTER_LINEAR);
  } else {working = grayscale;}
  cv::Mat gx, gy, magnitude;
  cv::Sobel(working, gx, CV_32F, 1, 0, 3);
  cv::Sobel(working, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, magnitude);
  std::vector<float> nonzero;
  nonzero.reserve(static_cast<std::size_t>(magnitude.total() / 4U));
  for (int row = 0; row < magnitude.rows; ++row) {
    const auto * values = magnitude.ptr<float>(row);
    for (int column = 0; column < magnitude.cols; ++column) {
      if (values[column] > std::numeric_limits<float>::epsilon()) {nonzero.push_back(values[column]);}
    }
  }
  if (nonzero.empty()) {return;}
  const std::size_t index = static_cast<std::size_t>(std::floor(
    (parameters_.edge_percentile / 100.0) * static_cast<double>(nonzero.size() - 1U)));
  std::nth_element(nonzero.begin(), nonzero.begin() + static_cast<std::ptrdiff_t>(index), nonzero.end());
  const float threshold = nonzero[index];
  if (!std::isfinite(threshold) || threshold <= std::numeric_limits<float>::epsilon()) {return;}
  cv::Mat edges;
  cv::compare(magnitude, threshold, edges, cv::CMP_GE);
  if (cv::countNonZero(edges) == 0) {return;}
  cv::bitwise_not(edges, edges);
  cv::distanceTransform(edges, distance_px_, cv::DIST_L2, 3);
  gradient_x_ = gx;
  gradient_y_ = gy;
}

bool RgbEdgePrior::ready() const
{
  return !distance_px_.empty();
}

core::VisualEvidence RgbEdgePrior::score(const core::Pose & pose) const
{
  if (!ready()) {return {};}
  const auto corners = corners_world(pose);
  std::vector<cv::Point3d> camera_points;
  camera_points.reserve(corners.size());
  for (const auto & corner : corners) {
    const Eigen::Vector3d local = world_from_camera_.linear().transpose() *
      (corner - world_from_camera_.translation());
    if (!local.allFinite() || local.z() <= 1e-6) {camera_points.emplace_back(0.0, 0.0, -1.0);}
    else {camera_points.emplace_back(local.x(), local.y(), local.z());}
  }
  std::vector<cv::Point2d> uv;
  cv::projectPoints(
    camera_points, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), camera_matrix_, distortion_, uv);
  const std::array<std::array<int, 2>, 12> canonical_edges{{{{0, 1}}, {{1, 3}}, {{3, 2}}, {{2, 0}}, {{4, 5}}, {{5, 7}}, {{7, 6}}, {{6, 4}}, {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}}};
  std::vector<cv::Point> samples;
  std::vector<cv::Point2d> normals;
  const auto signs = corner_signs();
  for (const auto & edge : canonical_edges) {
    const int start = edge[0], end = edge[1];
    if (camera_points[start].z <= 0.0 || camera_points[end].z <= 0.0 ||
      !std::isfinite(uv[start].x) || !std::isfinite(uv[start].y) ||
      !std::isfinite(uv[end].x) || !std::isfinite(uv[end].y)) {continue;}
    const Eigen::Vector3d midpoint = (corners[start] + corners[end]) * 0.5;
    bool visible = false;
    for (int axis = 0; axis < 3; ++axis) {
      if (signs[start][axis] != signs[end][axis]) {continue;}
      const Eigen::Vector3d normal = pose.rotation.col(axis) * static_cast<double>(signs[start][axis]);
      if (normal.dot(world_from_camera_.translation() - midpoint) > 0.0) {visible = true; break;}
    }
    if (!visible) {continue;}
    const cv::Point2d delta = uv[end] - uv[start];
    const double length = cv::norm(delta);
    if (!std::isfinite(length) || length < 1e-6) {continue;}
    const int count = std::max(2, static_cast<int>(std::ceil(length / parameters_.sample_spacing_px)) + 1);
    const cv::Point2d normal(-delta.y / length, delta.x / length);
    for (int index = 0; index < count; ++index) {
      const double fraction = static_cast<double>(index) / static_cast<double>(count - 1);
      samples.emplace_back(cvRound(uv[start].x + fraction * delta.x), cvRound(uv[start].y + fraction * delta.y));
      normals.push_back(normal);
    }
  }
  double sum = 0.0;
  std::size_t used = 0U, supported = 0U;
  const double sx = static_cast<double>(distance_px_.cols) / source_size_.width;
  const double sy = static_cast<double>(distance_px_.rows) / source_size_.height;
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto & pixel = samples[index];
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= source_size_.width || pixel.y >= source_size_.height ||
      (!excluded_mask_.empty() && excluded_mask_.at<std::uint8_t>(pixel.y, pixel.x) != 0U)) {continue;}
    const int u = std::clamp(cvRound(pixel.x * sx), 0, distance_px_.cols - 1);
    const int v = std::clamp(cvRound(pixel.y * sy), 0, distance_px_.rows - 1);
    const double distance = distance_px_.at<float>(v, u) / sx;
    const double gx = gradient_x_.at<float>(v, u), gy = gradient_y_.at<float>(v, u);
    const double magnitude = std::hypot(gx, gy);
    const double alignment = magnitude > 1e-9 ?
      std::abs((gx * normals[index].x + gy * normals[index].y) / magnitude) : 0.0;
    sum += std::exp(-distance / parameters_.distance_scale_px) * (0.75 + 0.25 * alignment);
    supported += distance <= parameters_.distance_scale_px ? 1U : 0U;
    ++used;
  }
  if (used < static_cast<std::size_t>(parameters_.min_samples) ||
    static_cast<double>(supported) / static_cast<double>(used) < parameters_.min_support_fraction) {return {};}
  return {parameters_.weight * sum / static_cast<double>(used), true};
}
}  // namespace blockpose::vision
