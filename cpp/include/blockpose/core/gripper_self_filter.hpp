#pragma once

// Small, deterministic point self-filter. It deliberately stays in the
// detector core rather than converting a cloud to and from PCL just to apply
// the two FK-defined gripper exclusion boxes.
#include "blockpose/core/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace blockpose::core
{

struct OrientedBox
{
  Point center{Point::Zero()};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  Point size{Point::Zero()};
};

inline void validate_oriented_box(const OrientedBox & box)
{
  if (!box.center.allFinite() || !box.rotation.allFinite() || !box.size.allFinite() ||
    box.size.x() <= 0.0 || box.size.y() <= 0.0 || box.size.z() <= 0.0) {
    throw std::invalid_argument("oriented box center, rotation, and positive size must be finite");
  }
  if (!box.rotation.transpose().isApprox(box.rotation.inverse(), 1e-6) ||
    std::abs(box.rotation.determinant() - 1.0) > 1e-6) {
    throw std::invalid_argument("oriented box rotation must be orthonormal");
  }
}

inline bool point_inside_oriented_box(const Point & point, const OrientedBox & box)
{
  const Point local = box.rotation.transpose() * (point - box.center);
  return (local.array().abs() <= (box.size * 0.5).array()).all();
}

inline Points remove_points_inside_oriented_boxes(
  const Points & input, const std::vector<OrientedBox> & boxes,
  std::size_t * removed_count = nullptr)
{
  for (const auto & box : boxes) {validate_oriented_box(box);}

  Points output;
  output.reserve(input.size());
  std::size_t removed = 0U;
  for (const auto & point : input) {
    const bool filtered = std::any_of(boxes.begin(), boxes.end(), [&point](const auto & box) {
        return point_inside_oriented_box(point, box);
      });
    if (filtered) {
      ++removed;
    } else {
      output.push_back(point);
    }
  }
  if (removed_count != nullptr) {
    *removed_count = removed;
  }
  return output;
}

}  // namespace blockpose::core
