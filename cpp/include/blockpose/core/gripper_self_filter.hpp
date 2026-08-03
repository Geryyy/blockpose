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

// Portable geometry for the PZS100 two-rail self filter.  Adapters resolve
// their own frame names (K9 -> K10_left_rail and K11 -> K12_right_rail in the
// production robot) and provide the resulting world-frame endpoints here.
// Keeping TF names out of this API makes the exact same filter usable from a
// ROS callback, a captured snapshot, and Python through nanobind.
struct Pzs100RailPose
{
  Point parent_position{Point::Zero()};
  Point rail_position{Point::Zero()};
  Eigen::Matrix3d world_from_parent_rotation{Eigen::Matrix3d::Identity()};
  Point outward_axis_parent{Point::UnitZ()};
};

struct Pzs100SelfFilterParameters
{
  // The local box axes are: parent->rail, outward from the gripper, and
  // across the rail.  The two boxes remove the material outside each rail,
  // not the block held between them.
  double outboard_extent_m{1.50};
  double cross_rail_extent_m{1.20};
  double rail_end_margin_m{0.15};
};

struct Pzs100GripperFilterBox
{
  OrientedBox box;
  Point rail_start{Point::Zero()};
  Point rail_end{Point::Zero()};
  Point outward_normal{Point::UnitZ()};
};

inline void validate_pzs100_self_filter_parameters(const Pzs100SelfFilterParameters & parameters)
{
  if (!std::isfinite(parameters.outboard_extent_m) ||
    !std::isfinite(parameters.cross_rail_extent_m) ||
    !std::isfinite(parameters.rail_end_margin_m) ||
    parameters.outboard_extent_m <= 0.0 || parameters.cross_rail_extent_m <= 0.0 ||
    parameters.rail_end_margin_m < 0.0)
  {
    throw std::invalid_argument("PZS100 self-filter extents must be finite, positive, and have non-negative rail margin");
  }
}

inline Pzs100GripperFilterBox make_pzs100_gripper_filter_box(
  const Pzs100RailPose & rail, const Pzs100SelfFilterParameters & parameters = {})
{
  validate_pzs100_self_filter_parameters(parameters);
  if (!rail.parent_position.allFinite() || !rail.rail_position.allFinite() ||
    !rail.world_from_parent_rotation.allFinite() || !rail.outward_axis_parent.allFinite() ||
    rail.outward_axis_parent.norm() <= 1e-9)
  {
    throw std::invalid_argument("PZS100 rail pose must contain finite endpoints, rotation, and outward axis");
  }
  if (!(rail.world_from_parent_rotation.transpose() * rail.world_from_parent_rotation).
    isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
    std::abs(rail.world_from_parent_rotation.determinant() - 1.0) > 1e-6)
  {
    throw std::invalid_argument("PZS100 parent rotation must be orthonormal");
  }
  const Point rail_axis = rail.rail_position - rail.parent_position;
  const double rail_length = rail_axis.norm();
  if (rail_length <= 1e-9) {throw std::invalid_argument("PZS100 rail endpoints coincide");}
  const Point rail_direction = rail_axis / rail_length;
  Point outward = rail.world_from_parent_rotation * rail.outward_axis_parent;
  outward -= rail_direction * outward.dot(rail_direction);
  const double outward_length = outward.norm();
  if (outward_length <= 1e-9) {
    throw std::invalid_argument("PZS100 outward axis is parallel to rail");
  }
  outward /= outward_length;
  Eigen::Matrix3d rotation;
  rotation.col(0) = rail_direction;
  rotation.col(1) = outward;
  rotation.col(2) = rail_direction.cross(outward).normalized();

  Pzs100GripperFilterBox result;
  result.rail_start = rail.parent_position;
  result.rail_end = rail.rail_position;
  result.outward_normal = outward;
  result.box.center = (rail.parent_position + rail.rail_position) * 0.5 +
    outward * (parameters.outboard_extent_m * 0.5);
  result.box.rotation = rotation;
  result.box.size = Point(
    rail_length + 2.0 * parameters.rail_end_margin_m,
    parameters.outboard_extent_m, parameters.cross_rail_extent_m);
  return result;
}

inline std::vector<Pzs100GripperFilterBox> make_pzs100_gripper_filter_boxes(
  const std::vector<Pzs100RailPose> & rails,
  const Pzs100SelfFilterParameters & parameters = {})
{
  validate_pzs100_self_filter_parameters(parameters);
  std::vector<Pzs100GripperFilterBox> result;
  result.reserve(rails.size());
  for (const auto & rail : rails) {
    result.push_back(make_pzs100_gripper_filter_box(rail, parameters));
  }
  return result;
}

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

inline Points keep_points_inside_axis_aligned_box(
  const Points & input, const Point & minimum, const Point & maximum,
  std::size_t * removed_count = nullptr)
{
  if (!minimum.allFinite() || !maximum.allFinite() || (minimum.array() > maximum.array()).any()) {
    throw std::invalid_argument("workspace minimum must not exceed finite maximum");
  }
  Points output; output.reserve(input.size()); std::size_t removed = 0U;
  for (const auto & point : input) {
    if ((point.array() >= minimum.array()).all() && (point.array() <= maximum.array()).all()) {
      output.push_back(point);
    } else {
      ++removed;
    }
  }
  if (removed_count != nullptr) {*removed_count = removed;}
  return output;
}

}  // namespace blockpose::core
