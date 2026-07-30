#pragma once

// Sensor-ray visibility and free-space evidence for cuboid hypotheses.
#include "blockpose/core/geometry.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace blockpose::core
{
struct SensorContext
{
  Point origin{Point::Zero()};
  Points ray_directions;
  std::vector<double> ranges;
  Points miss_directions;
  Points ray_origins;
  Points miss_origins;
};
struct VisibilityEvidence {std::size_t expected_faces{0}, covered_faces{0}, violations{0}, supported_rays{0}, incident_rays{0};};

inline SensorContext validate_sensor_context(SensorContext context)
{
  if (!context.origin.allFinite() || context.ray_directions.size() != context.ranges.size()) {throw std::invalid_argument("invalid SensorContext ray arrays");}
  if (context.ray_origins.empty()) {context.ray_origins.assign(context.ray_directions.size(), context.origin);}
  if (context.miss_origins.empty()) {context.miss_origins.assign(context.miss_directions.size(), context.origin);}
  if (context.ray_origins.size() != context.ray_directions.size() || context.miss_origins.size() != context.miss_directions.size()) {throw std::invalid_argument("SensorContext origins must match rays");}
  for (std::size_t i = 0; i < context.ray_directions.size(); ++i) {const double n = context.ray_directions[i].norm(); if (!context.ray_directions[i].allFinite() || !context.ray_origins[i].allFinite() || !std::isfinite(context.ranges[i]) || context.ranges[i] < 0.0 || n <= 1e-12) {throw std::invalid_argument("invalid SensorContext return ray");} context.ray_directions[i] /= n;}
  for (std::size_t i = 0; i < context.miss_directions.size(); ++i) {const double n = context.miss_directions[i].norm(); if (!context.miss_directions[i].allFinite() || !context.miss_origins[i].allFinite() || n <= 1e-12) {throw std::invalid_argument("invalid SensorContext miss ray");} context.miss_directions[i] /= n;}
  return context;
}

inline bool ray_box_interval(const Point & origin, const Point & direction, const Point & half, double * exit)
{
  double enter = -std::numeric_limits<double>::infinity(), leave = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(direction[axis]) <= 1e-12) {if (origin[axis] < -half[axis] || origin[axis] > half[axis]) {return false;} continue;}
    double low = (-half[axis] - origin[axis]) / direction[axis], high = (half[axis] - origin[axis]) / direction[axis]; if (low > high) {std::swap(low, high);} enter = std::max(enter, low); leave = std::min(leave, high);
  }
  *exit = leave; return leave > std::max(enter, 0.0);
}

inline VisibilityEvidence visibility_evidence(const Point & position, const Eigen::Matrix3d & rotation, const std::array<double, 3> & dims, const SensorContext & raw_context, double tolerance = .06)
{
  const SensorContext context = validate_sensor_context(raw_context); const Point half(dims[0] / 2., dims[1] / 2., dims[2] / 2.); const double radius_sq = (half.array() + tolerance).square().sum(); VisibilityEvidence out;
  std::vector<std::size_t> relevant;
  for (std::size_t i = 0; i < context.ray_directions.size(); ++i) {const Point center = position - context.ray_origins[i]; const double projection = context.ray_directions[i].dot(center); const double perpendicular_sq = std::max(center.squaredNorm() - projection * projection, 0.0); if (center.squaredNorm() <= radius_sq || (projection > 0.0 && perpendicular_sq <= radius_sq + 1e-12)) {relevant.push_back(i);}}
  for (int axis = 0; axis < 3; ++axis) for (double sign : {-1.0, 1.0}) {
    const double face = sign * half[axis]; bool expected = false, covered = false;
    for (const auto i : relevant) {
      const Point origin = rotation.transpose() * (context.ray_origins[i] - position), direction = rotation.transpose() * context.ray_directions[i];
      if (sign * (origin[axis] - face) <= 0.0 || std::abs(direction[axis]) <= 1e-9) {continue;} expected = true; const double distance = (face - origin[axis]) / direction[axis]; if (distance <= 0.0) {continue;} const Point hit = origin + distance * direction; bool rectangle = true; for (int other = 0; other < 3; ++other) if (other != axis && std::abs(hit[other]) > half[other] + tolerance) {rectangle = false;} if (!rectangle) {continue;} ++out.incident_rays; if (std::abs(context.ranges[i] - distance) <= tolerance) {++out.supported_rays; covered = true;}
    }
    if (expected) {++out.expected_faces;} if (covered) {++out.covered_faces;}
  }
  const Point shrunk = (half.array() - tolerance).max(1e-3); for (const auto i : relevant) {double exit = 0.; if (ray_box_interval(rotation.transpose() * (context.ray_origins[i] - position), rotation.transpose() * context.ray_directions[i], shrunk, &exit) && context.ranges[i] > exit + tolerance) {++out.violations;}}
  return out;
}

inline std::size_t free_space_violations_from_misses(const Point & position, const Eigen::Matrix3d & rotation, const std::array<double, 3> & dims, const SensorContext & raw_context, double tolerance = .06)
{
  const SensorContext context = validate_sensor_context(raw_context); const Point half = (Point(dims[0] / 2., dims[1] / 2., dims[2] / 2.).array() - tolerance).max(1e-3); std::size_t violations = 0;
  for (std::size_t i = 0; i < context.miss_directions.size(); ++i) {double exit = 0.; if (ray_box_interval(rotation.transpose() * (context.miss_origins[i] - position), rotation.transpose() * context.miss_directions[i], half, &exit)) {++violations;}}
  return violations;
}

inline std::array<double, 5> normalized_evidence_features(std::size_t support_points, double height_error, const VisibilityEvidence & evidence)
{
  const double total = static_cast<double>(evidence.supported_rays + evidence.violations);
  return {{std::min(static_cast<double>(support_points) / 400., 1.), std::max(0., 1. - height_error / .2), evidence.expected_faces ? static_cast<double>(evidence.covered_faces) / evidence.expected_faces : 0., total ? static_cast<double>(evidence.supported_rays) / total : 0., evidence.incident_rays ? static_cast<double>(evidence.supported_rays) / evidence.incident_rays : 0.}};
}
inline double normalized_evidence_score(const std::array<double, 5> & f) {return .25 * f[0] + .15 * f[1] + .20 * f[2] + .20 * f[3] + .20 * f[4];}
}  // namespace blockpose::core
