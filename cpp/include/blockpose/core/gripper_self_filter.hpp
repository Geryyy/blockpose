#pragma once

// Small, deterministic point self-filter. OrientedBox and its helpers now
// live in geometry.hpp: evidence.hpp needs them for known-occluder ray
// tests and must not depend on the gripper filter.
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
  // Everything structurally above the jaws (carrier, boom, arm) is not
  // covered by the two outward-facing rail boxes above, which only remove
  // material outboard of each rail.  Both default to 0.0 (feature off) so
  // existing profiles/behaviour are unchanged until an adapter supplies real
  // above-gripper geometry.
  double above_exclusion_height_m{0.0};
  double above_exclusion_margin_m{0.0};
  // How far OUTBOARD of the rail centreline the box starts. The rail frame
  // origin sits inboard of the jaw's actual contact face, so a box whose
  // inner face lies exactly on the centreline reaches past the jaw and eats
  // the gripped block: measured over 59 annotated blocks it removed 6.3% of
  // all their points, up to 55% on a single block, and cost 5 points of
  // detector recall outright. 100% of the wrongly removed points sat in the
  // inner half of this axis. Defaults to 0.0 so existing profiles keep their
  // current geometry; see config/pzs100_gripper_filter.yaml for the measured
  // production value.
  double inner_clearance_m{0.0};
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
  if (!std::isfinite(parameters.above_exclusion_height_m) ||
    !std::isfinite(parameters.above_exclusion_margin_m) ||
    parameters.above_exclusion_height_m < 0.0 || parameters.above_exclusion_margin_m < 0.0)
  {
    throw std::invalid_argument(
      "PZS100 above-gripper exclusion height and margin must be finite and non-negative");
  }
  if (!std::isfinite(parameters.inner_clearance_m) || parameters.inner_clearance_m < 0.0 ||
    parameters.inner_clearance_m >= parameters.outboard_extent_m)
  {
    throw std::invalid_argument(
      "PZS100 inner clearance must be finite, non-negative, and smaller than the outboard extent");
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

  // The box spans from `inner_clearance_m` outboard of the rail centreline
  // out to `outboard_extent_m`, so raising the clearance narrows it from the
  // inside only -- the outer reach, which is what actually masks the jaw, is
  // unchanged.
  const double thickness = parameters.outboard_extent_m - parameters.inner_clearance_m;
  Pzs100GripperFilterBox result;
  result.rail_start = rail.parent_position;
  result.rail_end = rail.rail_position;
  result.outward_normal = outward;
  result.box.center = (rail.parent_position + rail.rail_position) * 0.5 +
    outward * (parameters.inner_clearance_m + thickness * 0.5);
  result.box.rotation = rotation;
  result.box.size = Point(
    rail_length + 2.0 * parameters.rail_end_margin_m,
    thickness, parameters.cross_rail_extent_m);
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


// Builds the single oriented volume covering the jaw carrier, boom, and arm
// that sit structurally above both gripper jaws.  Unlike
// make_pzs100_gripper_filter_box, this is not a per-rail box: the boom rides
// in the gap *between* the two rails, a region the outward-facing per-rail
// boxes never reach, so one combined volume spanning both rails is required.
//
// "Up" is world +Z, and deliberately not any structural axis of the rig.
// An earlier version anchored it to a rail's own cross-axis (col(2)) on the
// theory that a structural direction survives rig tilt.  That was wrong twice
// over: it assumed a rail convention that does not hold -- when the rail runs
// vertically, col(2) comes out horizontal and an ordinary upright gripper was
// rejected -- and it solved the wrong problem.  Which side the payload is on
// is not structural but gravitational: the block is clamped between the jaws
// and hangs below them, while the carrier, boom and arm are above.  World +Z
// therefore separates the two under any rig orientation, and it carries no
// sign to resolve, so none of the tilt/inversion guards that convention
// needed are required here.  The volume's horizontal footprint is taken from
// the jaw boxes themselves, so it needs no assumption about which structural
// axis is which.
inline OrientedBox make_pzs100_above_gripper_exclusion_box(
  const std::vector<Pzs100RailPose> & rails, const Pzs100SelfFilterParameters & parameters = {})
{
  validate_pzs100_self_filter_parameters(parameters);
  if (rails.size() != 2) {
    throw std::invalid_argument("PZS100 above-gripper exclusion volume requires exactly two rails");
  }
  if (parameters.above_exclusion_height_m <= 0.0) {
    throw std::invalid_argument(
      "PZS100 above-gripper exclusion height must be positive to build a volume");
  }
  const auto box0 = make_pzs100_gripper_filter_box(rails[0], parameters);
  const auto box1 = make_pzs100_gripper_filter_box(rails[1], parameters);

  constexpr double kMinAxisComponent = 1e-6;
  const Point up = Point::UnitZ();

  // The horizontal frame comes from how the two jaws are separated, which is
  // the one thing a two-jaw gripper always has: they straddle the block, so
  // their centrelines are horizontally apart whatever the rails' own axis
  // convention happens to be.
  const Point rail_center_0 = (rails[0].parent_position + rails[0].rail_position) * 0.5;
  const Point rail_center_1 = (rails[1].parent_position + rails[1].rail_position) * 0.5;
  Point transverse = rail_center_1 - rail_center_0;
  transverse -= up * transverse.dot(up);
  const double transverse_norm = transverse.norm();
  if (transverse_norm <= kMinAxisComponent) {
    throw std::invalid_argument(
      "PZS100 rail centrelines are not horizontally separated -- cannot span the gap between "
      "the jaws");
  }
  transverse /= transverse_norm;
  // forward x transverse = up keeps the basis right-handed (det = +1).
  const Point forward = transverse.cross(up);

  // Footprint: the combined horizontal extent of both jaw boxes along each
  // axis, so the volume covers whatever the gripper actually occupies rather
  // than a reconstruction from rail geometry. The half-extent of an oriented
  // box along a direction is the sum of its per-axis projections.
  const auto half_extent = [](const OrientedBox & box, const Point & axis) {
      return 0.5 * (std::abs(axis.dot(box.rotation.col(0))) * box.size.x() +
             std::abs(axis.dot(box.rotation.col(1))) * box.size.y() +
             std::abs(axis.dot(box.rotation.col(2))) * box.size.z());
    };
  const auto span = [&](const Point & axis) {
      const double c0 = axis.dot(box0.box.center), h0 = half_extent(box0.box, axis);
      const double c1 = axis.dot(box1.box.center), h1 = half_extent(box1.box, axis);
      return std::pair<double, double>{
        std::min(c0 - h0, c1 - h1), std::max(c0 + h0, c1 + h1)};
    };
  const auto forward_span = span(forward);
  const auto transverse_span = span(transverse);
  const auto up_span = span(up);

  const double margin = parameters.above_exclusion_margin_m;
  const double length = forward_span.second - forward_span.first + margin;
  const double width = transverse_span.second - transverse_span.first + margin;
  if (length <= kMinAxisComponent || width <= kMinAxisComponent) {
    throw std::invalid_argument("PZS100 jaw boxes have no horizontal extent to span");
  }

  // Start at the highest point of either jaw box, so the volume sits strictly
  // above the gripper and can never reach the block clamped below it.
  const double top = up_span.second;
  const Point centre_forward = forward * (0.5 * (forward_span.first + forward_span.second));
  const Point centre_transverse = transverse * (0.5 * (transverse_span.first + transverse_span.second));

  OrientedBox result;
  result.rotation.col(0) = forward;
  result.rotation.col(1) = transverse;
  result.rotation.col(2) = up;
  result.size = Point(length, width, parameters.above_exclusion_height_m);
  result.center = centre_forward + centre_transverse +
    up * (top + parameters.above_exclusion_height_m * 0.5);
  validate_oriented_box(result);
  return result;
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
