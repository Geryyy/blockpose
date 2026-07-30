#pragma once

// Optional pose knowledge from outside geometric scene discovery.  This stays
// ROS-free so FK, RGB, a wall plan, and previously registered blocks all use
// exactly the same scoring path.

#include "blockpose/core/geometry.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace blockpose::core
{

struct PosePrior
{
  std::string source;
  Point position{Point::Zero()};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  std::array<double, 3> dims{{0.9, 0.6, 0.6}};
  // A zero weight is deliberately a no-op.  It lets callers ship a uniform
  // request schema even when a source is disabled or unavailable.
  double weight{0.0};
  double translation_tolerance_m{0.30};
  double orientation_tolerance_rad{0.70};
};

struct PriorMatch
{
  std::string source;
  double score{0.0};
  double translation_error_m{std::numeric_limits<double>::infinity()};
  double orientation_error_rad{std::numeric_limits<double>::infinity()};
};

using PosePriors = std::vector<PosePrior>;

inline Eigen::Matrix3d rotation_from_rpy(const Point & rpy)
{
  return Eigen::AngleAxisd(rpy.z(), Point::UnitZ()).toRotationMatrix() *
         Eigen::AngleAxisd(rpy.y(), Point::UnitY()).toRotationMatrix() *
         Eigen::AngleAxisd(rpy.x(), Point::UnitX()).toRotationMatrix();
}

inline double cuboid_orientation_error_rad(
  const Eigen::Matrix3d & rotation, const Eigen::Matrix3d & prior_rotation)
{
  // A rectangular cuboid is unchanged by a 180-degree turn about any local
  // axis.  Do not make a prior reject an otherwise identical physical pose
  // merely because the frame convention chose the opposite face direction.
  const std::array<Eigen::Vector3d, 4> signs{{
    Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d(1.0, -1.0, -1.0),
    Eigen::Vector3d(-1.0, 1.0, -1.0), Eigen::Vector3d(-1.0, -1.0, 1.0)}};
  double best = std::numeric_limits<double>::infinity();
  for (const auto & sign : signs) {
    const Eigen::Matrix3d symmetry = sign.asDiagonal();
    const Eigen::Matrix3d delta = (prior_rotation * symmetry).transpose() * rotation;
    const double cosine = std::clamp((delta.trace() - 1.0) * 0.5, -1.0, 1.0);
    best = std::min(best, std::acos(cosine));
  }
  return best;
}

inline PriorMatch best_prior_match(
  const Point & position, const Eigen::Matrix3d & rotation,
  const std::array<double, 3> & dims, const PosePriors * priors)
{
  PriorMatch best;
  if (priors == nullptr) {return best;}
  for (const auto & prior : *priors) {
    if (prior.weight <= 0.0 || !std::isfinite(prior.weight) ||
      prior.translation_tolerance_m <= 0.0 ||
      prior.orientation_tolerance_rad <= 0.0 ||
      !std::isfinite(prior.translation_tolerance_m) ||
      !std::isfinite(prior.orientation_tolerance_rad) ||
      prior.dims != dims || !prior.position.allFinite() ||
      !prior.rotation.allFinite())
    {
      continue;
    }
    const double translation = (position - prior.position).norm();
    const double orientation = cuboid_orientation_error_rad(rotation, prior.rotation);
    const double normalized_error =
      std::pow(translation / prior.translation_tolerance_m, 2.0) +
      std::pow(orientation / prior.orientation_tolerance_rad, 2.0);
    const double score = prior.weight * std::exp(-0.5 * normalized_error);
    if (score > best.score) {
      best = {prior.source, score, translation, orientation};
    }
  }
  return best;
}

}  // namespace blockpose::core
