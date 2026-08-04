#pragma once

// Pure detector-core geometry; deliberately no ROS or PCL dependency.
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blockpose::core
{
using Point = Eigen::Vector3d;
using Points = std::vector<Point>;

struct PlaneFitDiagnostics {
  std::size_t input_points{0}, search_points{0}, trials_evaluated{0}, valid_trials{0};
  // Search includes canonicalization/subsampling and RANSAC trial scoring;
  // full_score_refit covers only the all-point consensus and least-squares pass.
  double search_ms{0.0}, full_score_refit_ms{0.0};
};
struct PlaneModel {Point normal{Point::UnitZ()}; double offset{0.0};};
struct PlaneFitResult {PlaneModel plane; std::vector<std::size_t> inlier_indices; PlaneFitDiagnostics diagnostics;};
struct RansacParameters {double distance_threshold{0.02}; std::size_t num_iterations{1000}; std::uint64_t seed{0}; double confidence{0.9999}; std::size_t max_search_points{0};};

struct GroundPlane
{
  Point normal{Point::UnitZ()};
  double offset{0.0};
  std::string source{"empty"};
  [[nodiscard]] double height(const Point & p) const {return normal.dot(p) + offset;}
  [[nodiscard]] double support_z(const Point & p) const {return -(normal.x() * p.x() + normal.y() * p.y() + offset) / normal.z();}
};

struct LocalGroundModel : public GroundPlane
{
  double cell_size{0.5};
  std::map<std::pair<long long, long long>, double> cell_z;
  [[nodiscard]] double support_z(const Point & p) const
  {
    const double planar = GroundPlane::support_z(p);
    if (cell_z.empty()) {return planar;}
    const auto x = static_cast<long long>(std::floor(p.x() / cell_size));
    const auto y = static_cast<long long>(std::floor(p.y() / cell_size));
    const double max_distance_sq = std::pow(cell_size * 1.75, 2.0);
    const double not_found = std::numeric_limits<double>::infinity();
    double nearest_distance_sq = not_found;
    double nearest_z = planar;
    Point nearest_center = Point::Zero();
    // A support cell can be at most two cell indices away while remaining
    // within the 1.75-cell radius. This preserves the former nearest-cell
    // semantics without scanning every local-ground cell for every point.
    for (long long cell_x = x - 2; cell_x <= x + 2; ++cell_x) {
      for (long long cell_y = y - 2; cell_y <= y + 2; ++cell_y) {
        const auto candidate = cell_z.find({cell_x, cell_y});
        if (candidate == cell_z.end()) {continue;}
        const Point center((cell_x + .5) * cell_size, (cell_y + .5) * cell_size, 0.0);
        const double distance_sq = (p.head<2>() - center.head<2>()).squaredNorm();
        if (distance_sq < nearest_distance_sq) {
          nearest_distance_sq = distance_sq;
          nearest_z = candidate->second;
          nearest_center = center;
        }
      }
    }
    if (nearest_distance_sq > max_distance_sq) {return planar;}
    return planar + nearest_z - GroundPlane::support_z(nearest_center);
  }
  [[nodiscard]] double height(const Point & p) const {return p.z() - support_z(p);}
};

struct GroundParameters
{
  double z_percentile{5.0}, thickness{0.15}, ransac_distance{0.05}, normal_min_z{0.7};
  std::size_t ransac_iterations{500}; std::uint64_t ransac_seed{0};
  double local_cell_size{0.5}, local_clearance{0.05};
};
struct GroundRemovalResult {Points above_ground; LocalGroundModel ground;};

// A generic oriented box. Lives here rather than in gripper_self_filter.hpp
// because both the self-filter and the visibility evidence need it, and the
// latter must not depend on the former.
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

inline bool canonical_less(const Point & a, const Point & b)
{
  if (a.x() != b.x()) {return a.x() < b.x();}
  if (a.y() != b.y()) {return a.y() < b.y();}
  return a.z() < b.z();
}
inline Points canonical_order(Points p) {std::sort(p.begin(), p.end(), canonical_less); return p;}
inline double percentile_linear(std::vector<double> values, double percentile)
{
  if (values.empty() || !std::isfinite(percentile) || percentile < 0.0 || percentile > 100.0) {throw std::invalid_argument("invalid percentile");}
  std::sort(values.begin(), values.end()); const double idx = percentile / 100.0 * static_cast<double>(values.size() - 1U);
  const auto low = static_cast<std::size_t>(std::floor(idx)), high = static_cast<std::size_t>(std::ceil(idx));
  return values[low] + (idx - static_cast<double>(low)) * (values[high] - values[low]);
}
inline PlaneModel plane_from_triplet(const Point & a, const Point & b, const Point & c, bool * valid)
{
  Point normal = (b - a).cross(c - a); const double length = normal.norm();
  if (!std::isfinite(length) || length < 1e-12) {*valid = false; return {};}
  *valid = true; normal /= length; return {normal, -normal.dot(a)};
}

// RANSAC needs reproducible samples, not a particular Python RNG stream.
// Keep that requirement explicit with the standard engine and distribution.
class DeterministicSampler
{
public:
  explicit DeterministicSampler(std::uint64_t seed) : engine_(seed) {}

  [[nodiscard]] std::size_t index(std::size_t count)
  {
    if (count == 0U) {throw std::invalid_argument("sample count must be positive");}
    return std::uniform_int_distribution<std::size_t>(0U, count - 1U)(engine_);
  }

private:
  std::mt19937_64 engine_;
};

// Fit a dominant plane with deterministic RANSAC sampling.
inline PlaneFitResult segment_plane(const Points & input, const RansacParameters & params)
{
  if (input.size() < 3U) {throw std::invalid_argument("plane fitting needs at least 3 points");}
  if (!std::isfinite(params.distance_threshold) || params.distance_threshold <= 0.0 || params.num_iterations == 0U) {throw std::invalid_argument("invalid RANSAC parameters");}
  if (!std::isfinite(params.confidence) || params.confidence <= 0.0 || params.confidence >= 1.0) {throw std::invalid_argument("confidence must be in (0,1)");}
  const auto search_started = std::chrono::steady_clock::now();
  const Points points = canonical_order(input); Points search = points;
  if (params.max_search_points >= 3U && points.size() > params.max_search_points) {
    search.clear(); search.reserve(params.max_search_points);
    for (std::size_t i = 0; i < params.max_search_points; ++i) {
      search.push_back(points[static_cast<std::size_t>(std::floor(static_cast<double>(i) * (points.size() - 1U) / (params.max_search_points - 1U)))]);
    }
  }
  PlaneFitResult result; result.diagnostics.input_points = points.size(); result.diagnostics.search_points = search.size();
  DeterministicSampler generator(params.seed);
  std::size_t best_count = 0U, required = params.num_iterations;
  for (std::size_t trial = 0; trial < params.num_iterations && trial < required; ++trial) {
    const auto first = generator.index(search.size()), second = generator.index(search.size()), third = generator.index(search.size()); ++result.diagnostics.trials_evaluated;
    if (first == second || first == third || second == third) {continue;}
    bool valid = false; const PlaneModel candidate = plane_from_triplet(search[first], search[second], search[third], &valid);
    if (!valid) {continue;} ++result.diagnostics.valid_trials; std::size_t count = 0U;
    for (const auto & p : search) {if (std::abs(candidate.normal.dot(p) + candidate.offset) <= params.distance_threshold) {++count;}}
    if (count <= best_count) {continue;} best_count = count; result.plane = candidate;
    const double ratio = static_cast<double>(count) / static_cast<double>(search.size());
    if (ratio >= 1.0) {required = trial + 1U;} else if (ratio > 0.0) {
      const double denominator = std::log1p(-(ratio * ratio * ratio));
      if (denominator < 0.0) {required = std::min(params.num_iterations, trial + 1U + static_cast<std::size_t>(std::ceil(std::log1p(-params.confidence) / denominator)));}
    }
  }
  result.diagnostics.search_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - search_started).count();
  const auto full_score_refit_started = std::chrono::steady_clock::now();
  if (best_count < 3U) {throw std::runtime_error("RANSAC found no plane");}
  for (std::size_t i = 0; i < points.size(); ++i) {if (std::abs(result.plane.normal.dot(points[i]) + result.plane.offset) <= params.distance_threshold) {result.inlier_indices.push_back(i);}}
  if (result.inlier_indices.size() < 3U) {throw std::runtime_error("RANSAC full score has fewer than 3 inliers");}
  Point centroid = Point::Zero(); for (const auto i : result.inlier_indices) {centroid += points[i];} centroid /= static_cast<double>(result.inlier_indices.size());
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero(); for (const auto i : result.inlier_indices) {const Point d = points[i] - centroid; covariance.noalias() += d * d.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance); if (solver.info() != Eigen::Success) {throw std::runtime_error("least-squares plane refit failed");}
  Point normal = solver.eigenvectors().col(0); if (normal.dot(result.plane.normal) < 0.0) {normal = -normal;} const PlaneModel refit{normal, -normal.dot(centroid)};
  std::vector<std::size_t> refined; for (std::size_t i = 0; i < points.size(); ++i) {if (std::abs(refit.normal.dot(points[i]) + refit.offset) <= params.distance_threshold) {refined.push_back(i);}}
  if (refined.size() >= result.inlier_indices.size()) {result.plane = refit; result.inlier_indices = std::move(refined);}
  result.diagnostics.full_score_refit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - full_score_refit_started).count();
  return result;
}

inline LocalGroundModel make_local_ground_model(const Points & points, const GroundPlane & plane, const GroundParameters & params)
{
  LocalGroundModel local; local.normal = plane.normal; local.offset = plane.offset; local.source = plane.source; local.cell_size = params.local_cell_size;
  struct Sample {long long x, y; Point point;}; std::vector<Sample> samples; samples.reserve(points.size());
  for (const auto & p : points) {samples.push_back({static_cast<long long>(std::floor(p.x() / params.local_cell_size)), static_cast<long long>(std::floor(p.y() / params.local_cell_size)), p});}
  std::sort(samples.begin(), samples.end(), [](const Sample & a, const Sample & b) {return a.x == b.x ? a.y < b.y : a.x < b.x;});
  for (std::size_t begin = 0; begin < samples.size();) {
    std::size_t end = begin + 1U; while (end < samples.size() && samples[end].x == samples[begin].x && samples[end].y == samples[begin].y) {++end;}
    if (end - begin >= 3U) {std::vector<double> residuals; for (std::size_t i = begin; i < end; ++i) {residuals.push_back(samples[i].point.z() - plane.support_z(samples[i].point));}
      const double low = percentile_linear(residuals, 10.0); if (low <= params.thickness) {const Point center((samples[begin].x + .5) * params.local_cell_size, (samples[begin].y + .5) * params.local_cell_size, 0.0); local.cell_z.emplace(std::make_pair(samples[begin].x, samples[begin].y), plane.support_z(center) + low);}}
    begin = end;
  }
  return local;
}

inline GroundRemovalResult remove_ground(
  const Points & points, const GroundParameters & params,
  const Points * fitting_points = nullptr)
{
  if (!std::isfinite(params.local_cell_size) || params.local_cell_size <= 0.0) {throw std::invalid_argument("local_cell_size must be positive");}
  if (!std::isfinite(params.local_clearance) || params.local_clearance < 0.0) {throw std::invalid_argument("local_clearance must be non-negative");}
  if (points.empty()) {return {};}
  std::vector<double> zs; for (const auto & p : points) {zs.push_back(p.z());}
  const auto fallback = [&]() {GroundRemovalResult output; output.ground.normal = Point::UnitZ(); output.ground.offset = -percentile_linear(zs, params.z_percentile); output.ground.source = "percentile"; output.ground.cell_size = params.local_cell_size; for (const auto & p : points) {if (output.ground.height(p) >= params.thickness) {output.above_ground.push_back(p);}} return output;};
  const Points & fit_points = fitting_points == nullptr ? points : *fitting_points;
  if (fit_points.size() < 3U) {return fallback();}
  try {RansacParameters ransac; ransac.distance_threshold = params.ransac_distance; ransac.num_iterations = params.ransac_iterations; ransac.seed = params.ransac_seed; const auto fit = segment_plane(fit_points, ransac); PlaneModel model = fit.plane;
    if (std::abs(model.normal.z()) < params.normal_min_z) {return fallback();} if (model.normal.z() < 0.0) {model.normal = -model.normal; model.offset = -model.offset;}
    GroundRemovalResult output; output.ground = make_local_ground_model(points, GroundPlane{model.normal, model.offset, "ransac"}, params); for (const auto & p : points) {if (output.ground.height(p) >= params.local_clearance) {output.above_ground.push_back(p);}} return output;
  } catch (const std::exception &) {return fallback();}
}
}  // namespace blockpose::core
