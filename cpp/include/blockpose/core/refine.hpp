#pragma once

// Pure Eigen cuboid refinement; no ROS or PCL types.
#include "blockpose/core/geometry.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace blockpose::core
{
struct CuboidPose
{
  Point position{Point::Zero()};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};  // R_world_block
  std::array<double, 3> dims{{0.9, 0.6, 0.6}};
  double confidence{0.0};
  std::string source;
};
struct RefineParameters
{
  double huber_scale{0.05};
  double max_translation{0.15};
  double max_rotation_deg{20.0};
  std::size_t min_points{20};
  std::size_t max_evaluations{200};
};
inline Eigen::Matrix3d skew(const Point & v)
{
  Eigen::Matrix3d r;
  r << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return r;
}
inline Eigen::Matrix3d exp_so3(const Point & v)
{
  const double angle = v.norm();
  if (angle < 1.0e-12) {return Eigen::Matrix3d::Identity();}
  const Eigen::Matrix3d axis = skew(v / angle);
  return Eigen::Matrix3d::Identity() + std::sin(angle) * axis + (1.0 - std::cos(angle)) * axis * axis;
}
// Right Jacobian of SO(3): for R(w) = R0 * exp_so3(w), d(R(w)^T v)/dw =
// skew(R(w)^T v) * right_jacobian_so3(w). Needed because refine_pose's x.head<3>()
// is the *finite* right-multiplied rotation vector, not an infinitesimal
// perturbation, so the plain skew(q) term alone only matches a central-difference
// Jacobian at w == 0 (verified numerically; the two diverge away from w == 0).
inline Eigen::Matrix3d right_jacobian_so3(const Point & v)
{
  const double angle = v.norm();
  const Eigen::Matrix3d k = skew(v);
  if (angle < 1.0e-8) {return Eigen::Matrix3d::Identity() - 0.5 * k;}
  const double angle2 = angle * angle;
  return Eigen::Matrix3d::Identity() - ((1.0 - std::cos(angle)) / angle2) * k +
    ((angle - std::sin(angle)) / (angle2 * angle)) * k * k;
}
inline double box_sdf(const Point & local, const Point & half)
{
  const Point delta = local.cwiseAbs() - half;
  return delta.cwiseMax(Point::Zero()).norm() + std::min(delta.maxCoeff(), 0.0);
}
// box_sdf plus its gradient w.r.t. `local`. Exterior (some d_k > 0): gradient of
// the positive-part norm, zero on components already clipped to 0. Interior (all
// d_k <= 0, including the on-surface boundary d.maxCoeff() == 0): gradient of
// max_k d_k, a single nonzero component at the argmax. Both branches are
// subgradients at their shared boundary/kinks; any subgradient is valid there.
inline double box_sdf_gradient(const Point & local, const Point & half, Point * gradient)
{
  const Point delta = local.cwiseAbs() - half;
  const Point outside = delta.cwiseMax(Point::Zero());
  const double outside_norm = outside.norm();
  if (gradient != nullptr) {
    Point g = Point::Zero();
    if (delta.maxCoeff() > 0.0) {
      if (outside_norm > 0.0) {
        for (int k = 0; k < 3; ++k) {g[k] = (local[k] >= 0.0 ? 1.0 : -1.0) * outside[k] / outside_norm;}
      }
    } else {
      Eigen::Index k_max = 0; delta.maxCoeff(&k_max);
      g[k_max] = local[k_max] >= 0.0 ? 1.0 : -1.0;
    }
    *gradient = g;
  }
  return outside_norm + std::min(delta.maxCoeff(), 0.0);
}
inline std::vector<double> surface_distance(const CuboidPose & pose, const Points & points)
{
  const Point half(pose.dims[0] / 2.0, pose.dims[1] / 2.0, pose.dims[2] / 2.0);
  std::vector<double> result; result.reserve(points.size());
  for (const auto & point : points) {result.push_back(box_sdf(pose.rotation.transpose() * (point - pose.position), half));}
  return result;
}
inline std::vector<Points> assign_points(const std::vector<CuboidPose> & poses, const Points & points, double band)
{
  std::vector<Points> result(poses.size());
  for (const auto & point : points) {
    std::size_t owner = 0; double nearest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < poses.size(); ++i) {
      const Point half(poses[i].dims[0] / 2.0, poses[i].dims[1] / 2.0, poses[i].dims[2] / 2.0);
      const double distance = std::abs(box_sdf(poses[i].rotation.transpose() * (point - poses[i].position), half));
      if (distance < nearest) {nearest = distance; owner = i;}
    }
    if (!poses.empty() && nearest <= band) {result[owner].push_back(point);}
  }
  return result;
}
inline double huber_cost(double r, double scale)
{
  const double a = std::abs(r); return a <= scale ? 0.5 * r * r : scale * (a - 0.5 * scale);
}
inline double rotation_angle_deg(const Eigen::Matrix3d & r)
{
  return std::acos(std::clamp((r.trace() - 1.0) / 2.0, -1.0, 1.0)) * 57.2957795130823208768;
}
// Facts captured before the correction safety guard for diagnostics without
// changing the published pose or refinement decision.
struct RefineDiagnostics
{
  bool attempted{false};
  std::size_t point_count{0U};
  std::size_t evaluations{0U};
  std::size_t iterations{0U};
  double initial_cost{std::numeric_limits<double>::quiet_NaN()};
  double final_cost{std::numeric_limits<double>::quiet_NaN()};
  std::array<double, 6> correction{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  double translation_norm{0.0};
  double rotation_deg{0.0};
  bool guard_accepted{false};
};
struct RefineResult {CuboidPose pose; bool refined{false}; RefineDiagnostics diagnostics;};

inline RefineResult refine_pose(const CuboidPose & initial, const Points & points, const RefineParameters & params = {})
{
  RefineDiagnostics diagnostics;
  diagnostics.point_count = points.size();
  // The Jacobian is analytic now, so every LM trial (accepted or rejected)
  // costs exactly one residual evaluation; max_evaluations bounds that count
  // directly. Require at least the initial evaluation plus one trial.
  if (points.size() < params.min_points || params.huber_scale <= 0.0 || params.max_evaluations < 2U) {return {initial, false, diagnostics};}
  diagnostics.attempted = true;
  const Point half(initial.dims[0] / 2.0, initial.dims[1] / 2.0, initial.dims[2] / 2.0);
  const auto residuals = [&initial, &points, &half](const Eigen::Matrix<double, 6, 1> & x) {
      const Eigen::Matrix3d r = initial.rotation * exp_so3(x.head<3>());
      std::vector<double> output; output.reserve(points.size());
      for (const auto & point : points) {output.push_back(box_sdf(r.transpose() * (point - (initial.position + x.tail<3>())), half));}
      return output;
    };
  const auto cost_for = [&params](const std::vector<double> & values) {double total = 0.0; for (const double value : values) {total += huber_cost(value, params.huber_scale);} return total;};
  Eigen::Matrix<double, 6, 1> x = Eigen::Matrix<double, 6, 1>::Zero();
  std::vector<double> current = residuals(x); std::size_t evaluations = 1U; double cost = cost_for(current);
  diagnostics.initial_cost = cost;
  // Levenberg-Marquardt with Marquardt (diagonal-scaled) damping: (JtWJ +
  // lambda*diag(JtWJ)) delta = -JtW r. Accept a trial and shrink lambda when
  // it improves the Huber cost; reject and grow lambda otherwise. lambda
  // persists across outer iterations (re-scaled against each iteration's own
  // diagonal), which is the standard Marquardt scheme. Replaces the old fixed
  // 1e-8 ridge plus first-improvement backtracking line search.
  constexpr double kTau = 1.0e-3;
  constexpr double kLambdaDown = 1.0 / 3.0;
  constexpr double kLambdaUp = 2.0;
  constexpr double kLambdaMax = 1.0e12;
  constexpr double kLambdaMin = 1.0e-12;
  constexpr double kDiagFloor = 1.0e-12;
  constexpr double kStepTol = 1.0e-10;
  constexpr std::size_t kMaxIterations = 100U;
  constexpr std::size_t kMaxLambdaTrials = 30U;
  double lambda = -1.0;  // set from the first iteration's own diagonal scale
  bool keep_iterating = true;
  for (std::size_t iteration = 0; keep_iterating && iteration < kMaxIterations && evaluations < params.max_evaluations; ++iteration) {
    diagnostics.iterations = iteration + 1U;
    const Eigen::Matrix3d r = initial.rotation * exp_so3(x.head<3>());
    const Eigen::Matrix3d jr = right_jacobian_so3(x.head<3>());
    Eigen::MatrixXd jacobian(points.size(), 6);
    for (std::size_t row = 0; row < points.size(); ++row) {
      const Point q = r.transpose() * (points[row] - (initial.position + x.tail<3>()));
      Point gradient;
      box_sdf_gradient(q, half, &gradient);
      // J_row = g^T * [skew(q), -R^T], rotation block additionally chain-ruled
      // through the right Jacobian (see right_jacobian_so3 comment above).
      const Eigen::Matrix<double, 1, 3> rotation_block = (gradient.transpose() * skew(q)) * jr;
      const Eigen::Matrix<double, 1, 3> translation_block = -(gradient.transpose() * r.transpose());
      jacobian.block<1, 3>(static_cast<Eigen::Index>(row), 0) = rotation_block;
      jacobian.block<1, 3>(static_cast<Eigen::Index>(row), 3) = translation_block;
    }
    Eigen::MatrixXd weighted_jacobian = jacobian;
    Eigen::VectorXd weighted_residual(points.size());
    for (std::size_t row = 0; row < points.size(); ++row) {
      const double weight = std::abs(current[row]) <= params.huber_scale ? 1.0 : params.huber_scale / std::abs(current[row]);
      const double root_weight = std::sqrt(weight);
      weighted_jacobian.row(static_cast<Eigen::Index>(row)) *= root_weight;
      weighted_residual[static_cast<Eigen::Index>(row)] = root_weight * current[row];
    }
    const Eigen::Matrix<double, 6, 6> normal = weighted_jacobian.transpose() * weighted_jacobian;
    const Eigen::Matrix<double, 6, 1> gradient_vector = weighted_jacobian.transpose() * weighted_residual;
    const Eigen::Matrix<double, 6, 1> scale = normal.diagonal().cwiseMax(kDiagFloor);
    if (lambda < 0.0) {lambda = kTau * scale.maxCoeff();}
    bool accepted = false;
    for (std::size_t trial = 0; trial < kMaxLambdaTrials && lambda <= kLambdaMax && evaluations < params.max_evaluations; ++trial) {
      Eigen::Matrix<double, 6, 6> damped = normal; damped.diagonal() += lambda * scale;
      const Eigen::Matrix<double, 6, 1> delta = damped.ldlt().solve(-gradient_vector);
      if (!delta.allFinite()) {lambda = std::min(kLambdaMax, lambda * kLambdaUp); continue;}
      const auto proposal = x + delta; const auto candidate = residuals(proposal); ++evaluations;
      const double candidate_cost = cost_for(candidate);
      if (candidate_cost < cost) {
        const double step_norm = delta.norm(); const double improvement = cost - candidate_cost;
        x = proposal; current = candidate; cost = candidate_cost;
        lambda = std::max(kLambdaMin, lambda * kLambdaDown); accepted = true;
        if (step_norm < kStepTol || improvement <= kStepTol * std::max(1.0, cost)) {keep_iterating = false;}
        break;
      }
      lambda = std::min(kLambdaMax, lambda * kLambdaUp);
    }
    if (!accepted) {break;}
  }
  const Eigen::Matrix3d delta_rotation = exp_so3(x.head<3>());
  diagnostics.evaluations = evaluations;
  diagnostics.final_cost = cost;
  for (int index = 0; index < 6; ++index) {diagnostics.correction[static_cast<std::size_t>(index)] = x[index];}
  diagnostics.translation_norm = x.tail<3>().norm();
  diagnostics.rotation_deg = rotation_angle_deg(delta_rotation);
  if (diagnostics.translation_norm > params.max_translation || diagnostics.rotation_deg > params.max_rotation_deg) {return {initial, false, diagnostics};}
  CuboidPose refined = initial; refined.position += x.tail<3>(); refined.rotation *= delta_rotation; refined.source += "+sdf";
  diagnostics.guard_accepted = true;
  return {refined, true, diagnostics};
}
inline std::vector<CuboidPose> refine_poses(std::vector<CuboidPose> poses, const Points & points, double band = 0.10, std::size_t iterations = 2U, const RefineParameters & params = {})
{
  for (std::size_t iteration = 0; iteration < std::max<std::size_t>(iterations, 1U); ++iteration) {const auto assigned = assign_points(poses, points, band); for (std::size_t i = 0; i < poses.size(); ++i) {poses[i] = refine_pose(poses[i], assigned[i], params).pose;}}
  return poses;
}
}  // namespace blockpose::core
