#pragma once

// Proposal, plane, hypothesis and selection stages of the pure DetectFree core.
// This is deliberately independent of ROS/PCL so the adapter cannot diverge.
#include "blockpose/core/geometry.hpp"
#include "blockpose/core/evidence.hpp"
#include "blockpose/core/priors.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <optional>
#include <tuple>

namespace blockpose::core
{
struct DetectionParameters
{
  double voxel_size{0.04}, dbscan_eps{0.07};
  double ground_thickness{0.15}, ground_ransac_distance{0.05}, ground_normal_min_z{0.7};
  std::size_t ground_ransac_iterations{500};
  double local_ground_cell_size{0.5}, local_ground_clearance{0.05};
  std::size_t dbscan_min_points{5}, cluster_min_size{30}, cluster_max_size{50000};
  double cluster_min_extent_xy{0.3}, cluster_max_extent_xy{1.5}, region_max_extent_xy{3.0};
  double cluster_min_extent_z{0.3}, cluster_max_extent_z{3.5}, cluster_max_center_z{3.0};
  double ransac_distance{0.02}; std::size_t ransac_iterations{1000}, ransac_search_max_points{2048}; std::uint64_t ransac_seed{0};
  std::size_t max_planes{5}, min_inliers{40}; double top_plane_angle_deg{30.0}, side_plane_angle_deg{20.0};
  std::array<double, 3> block_dims{{0.9, 0.6, 0.6}}; std::vector<std::array<double, 3>> candidate_dims;
  double refine_band{0.10}; std::size_t refine_iterations{2}, refine_min_points{20};
  double refine_huber_scale{0.05}, refine_max_translation{0.15}, refine_max_rotation_deg{20.0};
  bool refine_preserve_top_axis_if_gravity_worsens{false};
  double min_score{0.0}, fk_seed_min_evidence_gain{0.15}; bool fit_wide_proposals{false}; std::size_t conflict_alternatives{1}, proposal_max_components{8};
};
struct PlanePatch {Point normal{Point::UnitZ()}; Point centroid{Point::Zero()}; Points points; double residual_mad{0.0};};
struct Pose {Point position{Point::Zero()}; Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()}; std::array<double, 3> dims{{0.9, 0.6, 0.6}}; double confidence{1.0};};
struct HypothesisEvidence {std::size_t support_points{0}; double top_height_error_m{0.0}, score{0.0}; std::size_t expected_visible_faces{0}, covered_visible_faces{0}, free_space_violations{0}, supported_rays{0}, observed_geometry_faces{0}, incident_rays{0};};
struct VisualEvidence {double score{0.0}; bool available{false};};
using VisualScorer = std::function<VisualEvidence(const Pose &)>;
// A provenance record is deliberately kept separate from the numerical
// hypothesis, making pre-refinement decisions inspectable without changing
// any score, threshold, or selection behaviour.
struct RawHypothesisLineage
{
  std::string id;
  std::size_t proposal_component{0}, region{0}, top_plane{0};
  std::optional<std::size_t> side_plane;
  Point top_normal{Point::UnitZ()}, top_centroid{Point::Zero()};
  std::optional<Point> side_normal, side_centroid;
  std::array<double, 3> candidate_dims{{0.9, 0.6, 0.6}};
  bool top_only{false}, accepted_to_raw{false}, passed_score_threshold{false};
  Pose synthesized_pose;
  HypothesisEvidence evidence;
  double top_support_height_m{0.0};
  // Diagnostic only. Dims siblings share one top patch and differ by 90 degrees,
  // so the patch's own observed in-plane extents are the only quantity that can
  // say which axis is the 0.9 m one; the per-hypothesis evidence cannot, because
  // both siblings explain the same returns. Recorded to test that, not consumed.
  double top_patch_extent_major_m{0.0}, top_patch_extent_minor_m{0.0};
  // Extent along the synthesized pose's own local X and Y, so a sibling can be
  // compared against the axis length it actually claims.
  double top_patch_extent_local_x_m{0.0}, top_patch_extent_local_y_m{0.0};
  std::size_t top_patch_inliers{0};
  std::string fate{"geometric_rejected"};
};

// In-plane extents of a fitted patch: robust (2nd..98th percentile) spans along
// the patch's own principal in-plane axes, and along two supplied world axes.
inline void patch_in_plane_extents(
  const PlanePatch & patch, const Point & local_x, const Point & local_y,
  double * major, double * minor, double * along_x, double * along_y)
{
  if (patch.points.size() < 3U) {return;}
  const Point normal = patch.normal.normalized();
  const Point center = patch.centroid;
  Point first = std::abs(normal.x()) > .9 ? Point::UnitY() : Point::UnitX();
  first = (first - first.dot(normal) * normal).normalized();
  const Point second = normal.cross(first);
  std::vector<double> u, v, x, y;
  u.reserve(patch.points.size()); v.reserve(patch.points.size());
  x.reserve(patch.points.size()); y.reserve(patch.points.size());
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
  for (const auto & point : patch.points) {
    const Point delta = point - center;
    const double a = delta.dot(first), b = delta.dot(second);
    u.push_back(a); v.push_back(b);
    x.push_back(delta.dot(local_x)); y.push_back(delta.dot(local_y));
    covariance.noalias() += Eigen::Vector2d(a, b) * Eigen::Vector2d(a, b).transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solve(covariance);
  const auto span = [](std::vector<double> values) {
      return percentile_linear(values, 98.0) - percentile_linear(std::move(values), 2.0);
    };
  if (solve.info() == Eigen::Success) {
    const Eigen::Vector2d major_axis = solve.eigenvectors().col(1);
    const Eigen::Vector2d minor_axis = solve.eigenvectors().col(0);
    std::vector<double> along_major, along_minor;
    along_major.reserve(u.size()); along_minor.reserve(u.size());
    for (std::size_t index = 0; index < u.size(); ++index) {
      along_major.push_back(major_axis.x() * u[index] + major_axis.y() * v[index]);
      along_minor.push_back(minor_axis.x() * u[index] + minor_axis.y() * v[index]);
    }
    *major = span(std::move(along_major));
    *minor = span(std::move(along_minor));
  }
  *along_x = span(std::move(x));
  *along_y = span(std::move(y));
}
struct CuboidHypothesis {
  Pose pose;
  HypothesisEvidence evidence;
  VisualEvidence visual_evidence;
  std::optional<double> proposal_scale_m;
  double support_height_m{0.0};
  std::optional<std::size_t> lineage_index;
  PriorMatch prior_match;
  // Provenance is diagnostic-only.  The numerical selector deliberately does
  // not inspect either value; they let an offline trace follow a candidate
  // through refinement and NMS without reconstructing identity from floats.
  std::string trace_id;
  std::string source{"plane_fit"};
};
// leftover_points/stop_reason are diagnostic only: recorded strictly after the
// extraction loop below decides to stop, never consulted by it. stop_reason is
// one of "ransac_exception", "ransac_below_min_inliers", "max_planes_reached",
// "remaining_below_min_inliers" (region never had min_inliers points to begin
// a search), or the sentinel "not_run" if fit_planes itself was never called.
struct PlaneFitCounts {std::size_t calls{0}, search_points{0}, full_points_scored{0}, trials_evaluated{0}, valid_trials{0}, leftover_points{0}; std::string stop_reason{"not_run"};};

// Swapped horizontal dimensions describe the same physical cuboid; rotate its
// local frame into the package's canonical 0.9 x 0.6 x 0.6 convention before
// refinement and non-maximum suppression.
inline Pose canonicalize_pose(Pose pose)
{
  constexpr double tolerance = 1.0e-9;
  if (std::abs(pose.dims[0] - 0.6) <= tolerance &&
    std::abs(pose.dims[1] - 0.9) <= tolerance &&
    std::abs(pose.dims[2] - 0.6) <= tolerance)
  {
    Eigen::Matrix3d local_swap;
    local_swap << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    pose.rotation *= local_swap;
    pose.dims = {{0.9, 0.6, 0.6}};
  }
  return pose;
}

inline std::array<double, 3> extent(const Points & points)
{ Point lower = points.empty() ? Point::Zero() : points.front(), upper = lower; for (const auto & p : points) {lower = lower.cwiseMin(p); upper = upper.cwiseMax(p);} const Point e = upper - lower; return {{e.x(), e.y(), e.z()}}; }
inline Point centroid(const Points & points)
{ Point result = Point::Zero(); for (const auto & p : points) {result += p;} return points.empty() ? result : result / static_cast<double>(points.size()); }
inline double median(std::vector<double> values) {return percentile_linear(std::move(values), 50.0);}

inline std::tuple<long long, long long, long long> spatial_cell(const Point & point, double cell_size)
{
  return std::make_tuple(
    static_cast<long long>(std::floor(point.x() / cell_size)),
    static_cast<long long>(std::floor(point.y() / cell_size)),
    static_cast<long long>(std::floor(point.z() / cell_size)));
}

inline std::map<std::tuple<long long, long long, long long>, std::vector<std::size_t>> make_spatial_grid(
  const Points & points, double cell_size)
{
  std::map<std::tuple<long long, long long, long long>, std::vector<std::size_t>> grid;
  for (std::size_t index = 0; index < points.size(); ++index) {grid[spatial_cell(points[index], cell_size)].push_back(index);}
  return grid;
}

inline std::vector<std::size_t> grid_neighbors(const Points & points,
  const std::map<std::tuple<long long, long long, long long>, std::vector<std::size_t>> & grid,
  std::size_t index, double cell_size, double radius_squared)
{
  const auto [x, y, z] = spatial_cell(points[index], cell_size);
  std::vector<std::size_t> result;
  for (long long dx = -1; dx <= 1; ++dx) for (long long dy = -1; dy <= 1; ++dy) for (long long dz = -1; dz <= 1; ++dz) {
    const auto bucket = grid.find(std::make_tuple(x + dx, y + dy, z + dz));
    if (bucket == grid.end()) {continue;}
    for (const auto other : bucket->second) {if ((points[index] - points[other]).squaredNorm() <= radius_squared) {result.push_back(other);}}
  }
  std::sort(result.begin(), result.end());
  return result;
}

// Diagnostic only: one entry per raw DBSCAN label (label order, before the
// compat-sort below reorders/truncates them), recording exactly which gate a
// component failed. gate_passed mirrors dbscan_proposals' own accept/reject
// decision; accepted_rank/truncated_by_max_components are only meaningful
// when gate_passed is true, since only gate-passing components are sorted
// and truncated to proposal_max_components.
struct ProposalComponentDiagnostics
{
  std::size_t dbscan_label{0}, point_count{0};
  std::array<double, 3> extent{{0.0, 0.0, 0.0}};
  Point centroid{Point::Zero()};
  double center_ground_height{0.0};
  bool size_gate_ok{false}, extent_xy_gate_ok{false}, extent_z_gate_ok{false}, center_z_gate_ok{false}, gate_passed{false};
  bool truncated_by_max_components{false};
  std::optional<std::size_t> accepted_rank;
};

// Pure clustering step of dbscan_proposals below, with none of its gating,
// sorting or truncation. Extracted so an offline diagnostic probe can inspect
// a raw (pre-gate, pre-truncation) component's actual points -- the compact
// ProposalComponentDiagnostics above deliberately omits them to stay cheap
// for a whole-corpus report -- without duplicating the clustering loop
// itself. dbscan_proposals calls this directly; behaviour is unchanged.
inline std::vector<Points> dbscan_label_components(const Points & input, const DetectionParameters & params)
{
  const Points points = canonical_order(input); const std::size_t count = points.size(); std::vector<int> labels(count, -2); const double radius_sq = params.dbscan_eps * params.dbscan_eps;
  const auto grid = make_spatial_grid(points, params.dbscan_eps);
  const auto neighbors = [&](std::size_t index) {return grid_neighbors(points, grid, index, params.dbscan_eps, radius_sq);};
  std::vector<std::size_t> enqueued_generation(count, 0); std::size_t generation = 0; int label = 0;
  for (std::size_t index = 0; index < count; ++index) {if (labels[index] != -2) {continue;} auto seeds = neighbors(index); if (seeds.size() < params.dbscan_min_points) {labels[index] = -1; continue;} if (++generation == 0U) {std::fill(enqueued_generation.begin(), enqueued_generation.end(), 0); ++generation;} for (const auto seed : seeds) {enqueued_generation[seed] = generation;} labels[index] = label;
    for (std::size_t cursor = 0; cursor < seeds.size(); ++cursor) {const auto candidate = seeds[cursor]; if (labels[candidate] == -1) {labels[candidate] = label;} if (labels[candidate] != -2) {continue;} labels[candidate] = label; const auto adjacent = neighbors(candidate); if (adjacent.size() >= params.dbscan_min_points) {for (const auto next : adjacent) {if (enqueued_generation[next] != generation) {seeds.push_back(next); enqueued_generation[next] = generation;}}}} ++label;}
  std::vector<Points> components(static_cast<std::size_t>(label)); for (std::size_t i = 0; i < count; ++i) {if (labels[i] >= 0) {components[static_cast<std::size_t>(labels[i])].push_back(points[i]);}}
  return components;
}

inline std::vector<Points> dbscan_proposals(
  const Points & input, const DetectionParameters & params, const GroundPlane * ground = nullptr,
  std::vector<ProposalComponentDiagnostics> * diagnostics_out = nullptr)
{
  std::vector<Points> components = dbscan_label_components(input, params);
  if (diagnostics_out != nullptr) {diagnostics_out->clear(); diagnostics_out->reserve(components.size());}
  std::vector<Points> accepted; std::vector<std::size_t> accepted_labels;
  for (std::size_t component_label = 0; component_label < components.size(); ++component_label) {
    auto & proposal = components[component_label]; const auto e = extent(proposal);
    const bool size_ok = proposal.size() >= params.cluster_min_size && proposal.size() <= params.cluster_max_size;
    const bool extent_xy_ok = e[0] >= params.cluster_min_extent_xy && e[0] <= params.region_max_extent_xy && e[1] >= params.cluster_min_extent_xy && e[1] <= params.region_max_extent_xy;
    const bool extent_z_ok = e[2] >= params.cluster_min_extent_z && e[2] <= params.cluster_max_extent_z;
    const bool valid = size_ok && extent_xy_ok && extent_z_ok;
    const Point actual_center = centroid(proposal);
    const double center_height = ground != nullptr ? ground->height(actual_center) : 0.0;
    const bool center_z_ok = ground == nullptr || center_height <= params.cluster_max_center_z;
    if (diagnostics_out != nullptr) {
      ProposalComponentDiagnostics diag; diag.dbscan_label = component_label; diag.point_count = proposal.size(); diag.extent = e; diag.centroid = actual_center; diag.center_ground_height = center_height;
      diag.size_gate_ok = size_ok; diag.extent_xy_gate_ok = extent_xy_ok; diag.extent_z_gate_ok = extent_z_ok; diag.center_z_gate_ok = center_z_ok; diag.gate_passed = valid && center_z_ok;
      diagnostics_out->push_back(std::move(diag));
    }
    if (!valid) {continue;} if (ground != nullptr && center_height > params.cluster_max_center_z) {continue;}
    accepted_labels.push_back(component_label); accepted.push_back(std::move(proposal));
  }
  std::vector<std::size_t> order(accepted.size()); for (std::size_t i = 0; i < order.size(); ++i) {order[i] = i;}
  std::sort(order.begin(), order.end(), [&params, &accepted](std::size_t lhs, std::size_t rhs) {const Points & a = accepted[lhs]; const Points & b = accepted[rhs]; if (a.size() != b.size()) {return a.size() > b.size();} const auto ea = extent(a), eb = extent(b); const auto compat = [&params](const std::array<double, 3> & e) {double value = 0.0; for (std::size_t axis = 0; axis < 2U; ++axis) {const double c = std::clamp(std::round(e[axis] / params.block_dims[axis]), 1.0, 8.0); value += std::abs(e[axis] - c * params.block_dims[axis]) / params.block_dims[axis];} const double height_count = std::clamp(std::round(e[2] / params.block_dims[2]), 1.0, 6.0); return value + 0.5 * std::abs(e[2] - height_count * params.block_dims[2]) / params.block_dims[2];}; const double ca = compat(ea), cb = compat(eb); return ca == cb ? canonical_less(centroid(a), centroid(b)) : ca < cb;});
  std::vector<Points> sorted; sorted.reserve(order.size()); for (const auto i : order) {sorted.push_back(std::move(accepted[i]));}
  if (diagnostics_out != nullptr) {for (std::size_t rank = 0; rank < order.size(); ++rank) {auto & diag = (*diagnostics_out)[accepted_labels[order[rank]]]; if (rank < params.proposal_max_components) {diag.accepted_rank = rank;} else {diag.truncated_by_max_components = true;}}}
  if (sorted.size() > params.proposal_max_components) {sorted.resize(params.proposal_max_components);}
  return sorted;
}

inline std::vector<PlanePatch> split_plane_patch(const PlanePatch & patch, const DetectionParameters & params)
{
  if (patch.points.size() < 2U * params.min_inliers) {return {patch};} Point normal = patch.normal.normalized(), seed = std::abs(normal.x()) > .9 ? Point::UnitY() : Point::UnitX(); Point first = (seed - seed.dot(normal) * normal).normalized(), second = normal.cross(first); const double radius = std::max(params.voxel_size * 2.5, params.dbscan_eps * 1.25), radius_sq = radius * radius; const auto count = patch.points.size(); std::vector<std::size_t> parent(count); for (std::size_t i = 0; i < count; ++i) {parent[i] = i;}
  const auto root = [&parent](std::size_t value) {std::size_t r = value; while (parent[r] != r) {r = parent[r];} while (parent[value] != value) {const auto next = parent[value]; parent[value] = r; value = next;} return r;};
  Points projected; projected.reserve(count); for (const auto & point : patch.points) {projected.emplace_back(point.dot(first), point.dot(second), 0.0);} const auto grid = make_spatial_grid(projected, radius); for (std::size_t i = 0; i < count; ++i) {for (const auto j : grid_neighbors(projected, grid, i, radius, radius_sq)) {if (j <= i) {continue;} const auto a = root(i), b = root(j); if (a != b) {parent[std::max(a, b)] = std::min(a, b);}}}
  std::map<std::size_t, Points> groups; for (std::size_t i = 0; i < count; ++i) {groups[root(i)].push_back(patch.points[i]);} std::size_t valid = 0; for (const auto & [_, group] : groups) {if (group.size() >= params.min_inliers) {++valid;}} if (valid < 2) {return {patch};} std::vector<PlanePatch> output{patch};
  for (const auto & [_, group] : groups) {if (group.size() < params.min_inliers) {continue;} const Point center = centroid(group); std::vector<double> residuals; for (const auto & point : group) {residuals.push_back((point - center).dot(normal));} const double middle = median(residuals); for (auto & value : residuals) {value = std::abs(value - middle);} output.push_back({normal, center, group, median(std::move(residuals))});} return output;
}

inline std::pair<std::vector<PlanePatch>, PlaneFitCounts> fit_planes(const Points & region, const DetectionParameters & params)
{
  Points remaining = canonical_order(region); std::vector<PlanePatch> output; PlaneFitCounts counts; std::size_t plane_index = 0;
  for (; plane_index < params.max_planes && remaining.size() >= params.min_inliers; ++plane_index) {RansacParameters config; config.distance_threshold = params.ransac_distance; config.num_iterations = params.ransac_iterations; config.seed = params.ransac_seed + plane_index; config.max_search_points = params.ransac_search_max_points; PlaneFitResult fitted; try {fitted = segment_plane(remaining, config);} catch (const std::exception &) {counts.stop_reason = "ransac_exception"; break;} ++counts.calls; counts.search_points += fitted.diagnostics.search_points; counts.full_points_scored += fitted.diagnostics.input_points; counts.trials_evaluated += fitted.diagnostics.trials_evaluated; counts.valid_trials += fitted.diagnostics.valid_trials; if (fitted.inlier_indices.size() < params.min_inliers) {counts.stop_reason = "ransac_below_min_inliers"; break;} Points inliers; std::vector<bool> retained(remaining.size(), true); for (const auto i : fitted.inlier_indices) {inliers.push_back(remaining[i]); retained[i] = false;} const Point normal = fitted.plane.normal.normalized(), center = centroid(inliers); std::vector<double> residuals; for (const auto & point : inliers) {residuals.push_back((point - center).dot(normal));} const double middle = median(residuals); for (auto & value : residuals) {value = std::abs(value - middle);} const auto split = split_plane_patch({normal, center, inliers, median(std::move(residuals))}, params); output.insert(output.end(), split.begin(), split.end()); Points next; for (std::size_t i = 0; i < remaining.size(); ++i) {if (retained[i]) {next.push_back(remaining[i]);}} remaining = std::move(next);}
  // Diagnostic only, below: the loop above already decided why it stopped;
  // this just names that decision. Natural loop exit (no explicit break above
  // already set stop_reason) means either max_planes was reached or the
  // remainder fell under min_inliers before another RANSAC call was tried.
  counts.leftover_points = remaining.size();
  if (counts.stop_reason == "not_run") {counts.stop_reason = plane_index >= params.max_planes ? "max_planes_reached" : "remaining_below_min_inliers";}
  return {output, counts};
}

// Diagnostic only: one entry per accepted proposal component that
// detect_without_refinement skips entirely -- before fit_planes is ever
// called on it -- because it is wider than a single block on some axis and
// split_connected_row could not find observable per-block boundaries to tile
// it. A component here produced zero plane_region/plane_patch/pairing/
// raw_lineage entries, for a reason none of those diagnostics can show.
struct WideProposalSkipDiagnostics
{
  std::size_t proposal_component{0}, point_count{0}, region_count{0};
  std::array<double, 3> extent{{0.0, 0.0, 0.0}};
};
// Diagnostic only: per-region summary of the fit_planes call above (why it
// stopped, how many region points never joined any patch).
struct PlaneRegionDiagnostics
{
  std::size_t proposal_component{0}, region{0}, plane_count{0}, leftover_points{0};
  std::string stop_reason;
};
// Diagnostic only: one entry per patch fit_planes/split_plane_patch produced,
// independent of whether candidate_plane_sets later classifies it as a
// top/side candidate at all.
struct PlanePatchDiagnostics
{
  std::size_t proposal_component{0}, region{0}, plane_index{0}, point_count{0};
  Point normal{Point::UnitZ()}, centroid{Point::Zero()};
  double residual_mad{0.0};
};

inline std::vector<std::array<double, 3>> candidate_dims(const DetectionParameters & params)
{ std::vector<std::array<double, 3>> output; const auto configured = params.candidate_dims.empty() ? std::vector<std::array<double, 3>>{params.block_dims} : params.candidate_dims; for (const auto dims : configured) {for (const auto variant : {dims, std::array<double, 3>{{dims[1], dims[0], dims[2]}}}) {if (std::find(output.begin(), output.end(), variant) == output.end()) {output.push_back(variant);}}} return output; }

// Diagnostic only: per-plane top/side classification, and for planes that
// qualify as a top candidate, every side candidate that was tried and which
// gate (if any) rejected it. Populated beside decisions candidate_plane_sets
// already makes below; nothing here feeds `output`.
struct PairingDiagnostics
{
  struct SideAttempt
  {
    std::size_t side_plane_index{0};
    double drop_m{0.0}, gap_m{0.0}, tangential_overlap_m{0.0};
    bool drop_range_ok{false}, support_band_ok{false}, tangential_overlap_ok{false}, gap_ok{false};
    bool selected{false};
  };
  std::size_t plane_index{0};
  bool is_top{false}, is_side{false};
  std::optional<std::size_t> chosen_side_plane_index;
  std::vector<SideAttempt> side_attempts;
};
// Diagnostic only: one candidate_plane_sets() call (one proposal/region/dims
// combination) worth of PairingDiagnostics, tagged with where it happened.
struct RegionPairingDiagnostics
{
  std::size_t proposal_component{0}, region{0}, dims_index{0};
  std::vector<PairingDiagnostics> planes;
};

inline std::vector<std::pair<const PlanePatch *, const PlanePatch *>> candidate_plane_sets(
  const std::vector<PlanePatch> & planes, const DetectionParameters & params, const GroundPlane & ground,
  const std::array<double, 3> & dims, std::vector<PairingDiagnostics> * diagnostics_out = nullptr)
{
  const Point up = ground.normal.normalized(); const double top_cos = std::cos(params.top_plane_angle_deg * M_PI / 180.0), side_sin = std::sin(params.side_plane_angle_deg * M_PI / 180.0); std::vector<const PlanePatch *> tops, sides;
  if (diagnostics_out != nullptr) {diagnostics_out->clear(); diagnostics_out->resize(planes.size());}
  for (std::size_t plane_index = 0; plane_index < planes.size(); ++plane_index) {const auto & patch = planes[plane_index]; const bool is_top = std::abs(patch.normal.dot(up)) > top_cos; const bool is_side = std::abs(patch.normal.dot(up)) < side_sin; if (is_top) {tops.push_back(&patch);} if (is_side) {sides.push_back(&patch);} if (diagnostics_out != nullptr) {(*diagnostics_out)[plane_index].plane_index = plane_index; (*diagnostics_out)[plane_index].is_top = is_top; (*diagnostics_out)[plane_index].is_side = is_side;}}
  std::sort(tops.begin(), tops.end(), [&up](const auto * a, const auto * b) {return a->centroid.dot(up) > b->centroid.dot(up);}); std::vector<std::pair<const PlanePatch *, const PlanePatch *>> output;
  for (const auto * top : tops) {
    const PlanePatch * best = nullptr; double best_score = std::numeric_limits<double>::infinity(), top_height = top->centroid.dot(up);
    const std::size_t top_index = static_cast<std::size_t>(top - planes.data());
    for (const auto * side : sides) {
      Point n = side->normal - side->normal.dot(up) * up; if (n.norm() < 1e-8) {continue;} n.normalize();
      double low = std::numeric_limits<double>::infinity(), high = -low; for (const auto & point : side->points) {const double h = point.dot(up); low = std::min(low, h); high = std::max(high, h);}
      const double drop = top_height - side->centroid.dot(up);
      const bool drop_range_ok = !(drop < .15 || drop > dims[2] - .1), support_band_ok = !(high < top_height - .15 || low > top_height - .3);
      if (!drop_range_ok || !support_band_ok) {
        if (diagnostics_out != nullptr) {const std::size_t side_index = static_cast<std::size_t>(side - planes.data()); (*diagnostics_out)[top_index].side_attempts.push_back({side_index, drop, 0.0, 0.0, drop_range_ok, support_band_ok, false, false, false});}
        continue;
      }
      double gap = std::numeric_limits<double>::infinity(); for (const auto & point : top->points) {gap = std::min(gap, std::abs((point - side->centroid).dot(n)));}
      const Point tangent = up.cross(n).normalized(); double top_low = std::numeric_limits<double>::infinity(), top_high = -top_low, side_low = std::numeric_limits<double>::infinity(), side_high = -side_low;
      for (const auto & point : top->points) {const double projection = point.dot(tangent); top_low = std::min(top_low, projection); top_high = std::max(top_high, projection);}
      for (const auto & point : side->points) {const double projection = point.dot(tangent); side_low = std::min(side_low, projection); side_high = std::max(side_high, projection);}
      const double overlap = std::min(top_high, side_high) - std::max(top_low, side_low); const bool overlap_ok = overlap >= .15;
      if (!overlap_ok) {
        if (diagnostics_out != nullptr) {const std::size_t side_index = static_cast<std::size_t>(side - planes.data()); (*diagnostics_out)[top_index].side_attempts.push_back({side_index, drop, gap, overlap, drop_range_ok, support_band_ok, false, false, false});}
        continue;
      }
      const double score = gap + std::abs(drop - dims[2] / 2.0); const bool gap_ok = gap <= .1;
      if (diagnostics_out != nullptr) {const std::size_t side_index = static_cast<std::size_t>(side - planes.data()); (*diagnostics_out)[top_index].side_attempts.push_back({side_index, drop, gap, overlap, drop_range_ok, support_band_ok, overlap_ok, gap_ok, false});}
      if (gap_ok && (score < best_score || (score == best_score && best != nullptr && canonical_less(side->centroid, best->centroid)))) {best = side; best_score = score;}
    }
    if (diagnostics_out != nullptr && best != nullptr) {const std::size_t best_side_index = static_cast<std::size_t>(best - planes.data()); (*diagnostics_out)[top_index].chosen_side_plane_index = best_side_index; for (auto & attempt : (*diagnostics_out)[top_index].side_attempts) {if (attempt.side_plane_index == best_side_index) {attempt.selected = true;}}}
    output.emplace_back(top, best);
  }
  return output;
}

// The top-only fallback yaw is the dominant axis of the world-XY footprint.
// Taking it from the scatter projected onto the top plane is only equivalent
// when the fitted normal is exactly world +Z; a real stacked-block cluster
// with vertical extent differed by 34 degrees of yaw.
inline Pose synthesize_pose(const PlanePatch & top, const PlanePatch * side, const Points & cluster, const std::array<double, 3> & dims)
{
  Point z = top.normal.normalized(); if (z.z() < 0.0) {z = -z;} Point x; if (side != nullptr) {x = side->normal - side->normal.dot(z) * z; if (x.norm() < 1e-8) {x = Point::UnitX() - Point::UnitX().dot(z) * z;} x.normalize(); if ((centroid(cluster) - side->centroid).dot(x) < 0.0) {x = -x;}} else {Eigen::Vector2d center = Eigen::Vector2d::Zero(); for (const auto & point : cluster) {center += point.head<2>();} if (!cluster.empty()) {center /= static_cast<double>(cluster.size());} Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero(); for (const auto & point : cluster) {const Eigen::Vector2d offset = point.head<2>() - center; covariance.noalias() += offset * offset.transpose();} Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solve(covariance); const Eigen::Vector2d dominant = solve.info() == Eigen::Success ? Eigen::Vector2d(solve.eigenvectors().col(1)) : Eigen::Vector2d::UnitX(); x = Point(dominant.x(), dominant.y(), 0.0); if (x.norm() < 1e-8) {x = Point::UnitX();} else {x.normalize();} x -= x.dot(z) * z; if (x.norm() < 1e-8) {x = Point::UnitX() - Point::UnitX().dot(z) * z;} x.normalize();} Point y = z.cross(x).normalized(); Pose pose; pose.rotation.col(0) = x; pose.rotation.col(1) = y; pose.rotation.col(2) = z; pose.position = top.centroid - z * (dims[2] / 2.0); pose.dims = dims; pose.confidence = side == nullptr ? .5 : 1.; if (side != nullptr) {pose.position += x * (dims[0] / 2.0 - (pose.position - side->centroid).dot(x));} return canonicalize_pose(pose);
}

template<typename GroundModel>
inline CuboidHypothesis make_hypothesis(const Pose & pose, std::size_t support_points, const GroundModel & ground, std::size_t observed_faces, const SensorContext * sensor_context = nullptr)
{ const Point bottom = pose.position - pose.rotation.col(2) * (pose.dims[2] / 2.0); HypothesisEvidence evidence; evidence.support_points = support_points; evidence.top_height_error_m = std::abs(ground.height(bottom)); evidence.observed_geometry_faces = observed_faces; VisibilityEvidence visibility; if (sensor_context != nullptr) {visibility = visibility_evidence(pose.position, pose.rotation, pose.dims, *sensor_context); visibility.violations += free_space_violations_from_misses(pose.position, pose.rotation, pose.dims, *sensor_context); evidence.expected_visible_faces = visibility.expected_faces; evidence.covered_visible_faces = visibility.covered_faces; evidence.free_space_violations = visibility.violations; evidence.supported_rays = visibility.supported_rays; evidence.incident_rays = visibility.incident_rays;} evidence.score = normalized_evidence_score(normalized_evidence_features(evidence.support_points, evidence.top_height_error_m, visibility)); CuboidHypothesis result; result.pose = pose; result.evidence = evidence; result.support_height_m = ground.height(bottom); return result; }

inline bool boxes_overlap(const Pose & first, const Pose & second, double shrink = .08)
{
  Eigen::Vector3d a, b; for (int i = 0; i < 3; ++i) {a[i] = std::max(first.dims[i] / 2. - shrink, 1e-3); b[i] = std::max(second.dims[i] / 2. - shrink, 1e-3);} const Eigen::Vector3d relative = first.rotation.transpose() * (second.position - first.position); const Eigen::Matrix3d coupling = first.rotation.transpose() * second.rotation, absolute = coupling.cwiseAbs().array() + 1e-9; for (int axis = 0; axis < 3; ++axis) {if (std::abs(relative[axis]) > a[axis] + absolute.row(axis).dot(b)) {return false;}} for (int axis = 0; axis < 3; ++axis) {if (std::abs(relative.dot(coupling.col(axis))) > absolute.col(axis).dot(a) + b[axis]) {return false;}} for (int i = 0; i < 3; ++i) {for (int j = 0; j < 3; ++j) {const double ra = a[(i + 1) % 3] * absolute((i + 2) % 3, j) + a[(i + 2) % 3] * absolute((i + 1) % 3, j), rb = b[(j + 1) % 3] * absolute(i, (j + 2) % 3) + b[(j + 2) % 3] * absolute(i, (j + 1) % 3), distance = std::abs(relative[(i + 2) % 3] * coupling((i + 1) % 3, j) - relative[(i + 1) % 3] * coupling((i + 2) % 3, j)); if (distance > ra + rb) {return false;}}} return true;
}
inline double hypothesis_selection_score(const CuboidHypothesis & hypothesis)
{ return hypothesis.evidence.score + hypothesis.prior_match.score + (hypothesis.visual_evidence.available && std::isfinite(hypothesis.visual_evidence.score) ? hypothesis.visual_evidence.score : 0.0); }

inline std::vector<CuboidHypothesis> select_hypotheses(std::vector<CuboidHypothesis> hypotheses, double shrink = .08)
{ std::sort(hypotheses.begin(), hypotheses.end(), [](const auto & a, const auto & b) {const double a_score = hypothesis_selection_score(a), b_score = hypothesis_selection_score(b); return a_score == b_score ? canonical_less(a.pose.position, b.pose.position) : a_score > b_score;}); std::vector<CuboidHypothesis> selected; for (const auto & candidate : hypotheses) {bool conflict = false; for (const auto & prior : selected) {if (boxes_overlap(candidate.pose, prior.pose, shrink)) {conflict = true; break;}} if (!conflict) {selected.push_back(candidate);}} return selected; }

inline double pose_rotation_error_deg(const Pose & first, const Pose & second)
{
  const Eigen::Matrix3d delta = first.rotation.transpose() * second.rotation;
  return std::acos(std::clamp((delta.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
}

// Keeping more than one diverse seed per overlap group lets refinement choose
// between competing closed-form plane fits; final greedy NMS returns one pose
// per group.
inline std::vector<CuboidHypothesis> select_conflict_alternatives(
  std::vector<CuboidHypothesis> hypotheses, std::size_t limit, double shrink = .08)
{
  if (limit < 1U) {throw std::invalid_argument("conflict_alternatives must be at least one");}
  std::sort(hypotheses.begin(), hypotheses.end(), [](const auto & a, const auto & b) {
    const double a_score = hypothesis_selection_score(a), b_score = hypothesis_selection_score(b);
    return a_score == b_score ? canonical_less(a.pose.position, b.pose.position) : a_score > b_score;
  });
  std::vector<std::vector<CuboidHypothesis>> groups;
  for (const auto & candidate : hypotheses) {
    std::vector<std::size_t> overlapping;
    for (std::size_t index = 0; index < groups.size(); ++index) {
      if (std::any_of(groups[index].begin(), groups[index].end(), [&candidate, shrink](const auto & member) {
          return boxes_overlap(candidate.pose, member.pose, shrink);
        })) {overlapping.push_back(index);}
    }
    if (overlapping.empty()) {groups.push_back({candidate}); continue;}
    const std::size_t destination = overlapping.front();
    groups[destination].push_back(candidate);
    for (auto iterator = overlapping.rbegin(); iterator != overlapping.rend(); ++iterator) {
      if (*iterator == destination) {continue;}
      groups[destination].insert(groups[destination].end(), groups[*iterator].begin(), groups[*iterator].end());
      groups.erase(groups.begin() + static_cast<std::ptrdiff_t>(*iterator));
    }
  }
  std::vector<CuboidHypothesis> selected;
  for (const auto & group : groups) {
    std::vector<CuboidHypothesis> diverse;
    for (const auto & candidate : group) {
      const bool duplicate = std::any_of(diverse.begin(), diverse.end(), [&candidate](const auto & prior) {
          return (candidate.pose.position - prior.pose.position).norm() < .02 &&
                 pose_rotation_error_deg(candidate.pose, prior.pose) < 2.0;
        });
      if (!duplicate) {diverse.push_back(candidate);}
      if (diverse.size() == limit) {break;}
    }
    selected.insert(selected.end(), diverse.begin(), diverse.end());
  }
  return selected;
}
}  // namespace blockpose::core
