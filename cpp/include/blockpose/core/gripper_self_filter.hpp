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
  // Everything structurally above the jaws (carrier, boom, arm) is not
  // covered by the two outward-facing rail boxes above, which only remove
  // material outboard of each rail.  Both default to 0.0 (feature off) so
  // existing profiles/behaviour are unchanged until an adapter supplies real
  // above-gripper geometry.
  double above_exclusion_height_m{0.0};
  double above_exclusion_margin_m{0.0};
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

// Builds the single oriented volume covering the jaw carrier, boom, and arm
// that sit structurally above both gripper jaws.  Unlike
// make_pzs100_gripper_filter_box, this is not a per-rail box: the boom rides
// in the gap *between* the two rails, a region the outward-facing per-rail
// boxes never reach, so one combined volume spanning both rails is required.
//
// "Up" is anchored to the FIRST rail's own structural cross-axis (col(2)),
// which rotates with the rig, and is then sanity-checked against world +Z
// with a signed test.  See the body for why a sign taken from world +Z is
// unrecoverably ambiguous rather than merely imprecise.  A rig that is
// steeply tilted, inverted, or whose rails arrive in the opposite order is
// REFUSED: the failure mode is "no filter", never "filter inverted onto the
// payload".  The principled long-term fix is an explicit `up_axis_parent`
// in Pzs100RailPose, analogous to the existing `outward_axis_parent`, so up
// comes from rig structure with no world-frame assumption at all; that is
// not implemented here.
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
  // Anchor "up" to the FIRST rail's own cross-axis, which is a structural
  // direction that rotates with the rig, rather than to whichever sign
  // happens to point along world +Z.  Taking the sign from world +Z is not
  // merely imprecise, it is unrecoverably ambiguous: the two rails' cross
  // axes are antiparallel by construction (left rail outward=+Y gives
  // col(2)=+Z, right rail outward=-Y gives col(2)=-Z), so the pair {+Z,-Z}
  // looks identical whether the rig is upright or fully inverted, and a
  // sign chosen from |dot| admits BOTH bands -- at 135 deg of rig tilt it
  // resolves structural down and extrudes the exclusion volume through the
  // payload.  Anchoring to rails[0] removes that ambiguity entirely.
  //
  // The remaining assumption is the convention that rails[0]'s cross-axis
  // points structurally up.  That is checked below against world +Z with a
  // SIGNED test, so a rig that is inverted, lying on its side, or supplied
  // with its rails in the opposite order is REFUSED rather than silently
  // filtered the wrong way round -- the failure mode is "no filter", which
  // the caller surfaces as a visible disabled reason.
  constexpr double kMinUprightAlignment = 0.5;  // 60 deg from vertical
  const Point up = box0.box.rotation.col(2);
  if (up.dot(Point::UnitZ()) < kMinUprightAlignment) {
    throw std::invalid_argument(
      "PZS100 first rail's cross-axis is not within 60 degrees of world +Z -- the rig is "
      "tilted, inverted, or its rails are supplied in the wrong order; refusing to build "
      "the above-gripper exclusion volume rather than risk inverting it onto the payload");
  }
  // The second rail must oppose the first, confirming the pair really is a
  // gripper's two jaws rather than an unrelated pose supplied by mistake.
  if (box1.box.rotation.col(2).dot(up) > -kMinUprightAlignment) {
    throw std::invalid_argument("PZS100 rails disagree on which direction is up");
  }
  // `up` is a column of an already-orthonormal rotation, so it needs no
  // renormalisation here.

  // The forward (along-rail) axis of the combined volume: the average of
  // both rails' own forward axes, re-orthogonalised against up so the three
  // resulting columns stay an orthonormal, right-handed basis.
  Point forward = box0.box.rotation.col(0) + box1.box.rotation.col(0);
  forward -= up * forward.dot(up);
  const double forward_norm = forward.norm();
  if (forward_norm <= kMinAxisComponent) {
    throw std::invalid_argument("PZS100 rail directions are degenerate relative to up");
  }
  forward /= forward_norm;
  const Point transverse = up.cross(forward);  // forward x transverse = up (right-handed)

  // Span the true gap between the rails -- the rail centrelines themselves,
  // not the per-rail boxes (which are already offset outward, away from that
  // gap) -- plus the configured lateral margin.
  const Point rail_center_0 = (rails[0].parent_position + rails[0].rail_position) * 0.5;
  const Point rail_center_1 = (rails[1].parent_position + rails[1].rail_position) * 0.5;
  const double width = std::abs(transverse.dot(rail_center_1 - rail_center_0)) +
    parameters.above_exclusion_margin_m;
  if (width <= kMinAxisComponent) {
    throw std::invalid_argument("PZS100 rails coincide -- cannot span a width between them");
  }
  const double length = std::max(box0.box.size.x(), box1.box.size.x());

  // Extrude upward starting at the higher of the two jaw-box tops, so the
  // volume never dips down into either jaw box (and therefore never into the
  // payload hanging below them).
  const double top0 = up.dot(box0.box.center) + box0.box.size.z() * 0.5;
  const double top1 = up.dot(box1.box.center) + box1.box.size.z() * 0.5;
  const double top_height = std::max(top0, top1);
  const Point reference = (rail_center_0 + rail_center_1) * 0.5;

  OrientedBox result;
  result.rotation.col(0) = forward;
  result.rotation.col(1) = transverse;
  result.rotation.col(2) = up;
  result.size = Point(length, width, parameters.above_exclusion_height_m);
  result.center = reference +
    up * (top_height + parameters.above_exclusion_height_m * 0.5 - up.dot(reference));
  validate_oriented_box(result);
  return result;
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
