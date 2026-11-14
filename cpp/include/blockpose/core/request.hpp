#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <blockpose/core/pipeline.hpp>
#include <blockpose/core/priors.hpp>
#include <blockpose/core/refine.hpp>

// The detector request path, shared by the ROS node, the snapshot runner and the Python
// bench so a replayed request runs the code the crane ran.

namespace blockpose::core
{

// What a request prior may do beyond resolving ambiguity among cuboid hypotheses is the
// caller's decision, not something the detector reads off a label -- the core gates its two
// recovery paths on PosePrior::may_seed_refinement and ::may_anchor_end_face. PosePrior.msg
// carries no flag field (its interfaces package is frozen), so the request states them as
// prefixes on `source`, in this order, with the real label following:
//
//   "seed+registered_block:block_0"           seeds where the cloud proposed nothing
//   "endface+registered_block:block_2"        its yaw may anchor an end-face-only cluster
//   "seed+endface+registered_block:block_0"   both
//   "fk"                                      plain label, no capability
//
// Keep in step with the same two constants in concrete_block_world_model's perception
// orchestrator; nothing but a Gazebo run notices if the two halves drift. 331
constexpr std::string_view kSeedCapabilityPrefix = "seed+";
constexpr std::string_view kEndFaceCapabilityPrefix = "endface+";

inline bool strip_prefix(std::string & value, std::string_view prefix)
{
  if (value.rfind(prefix, 0U) != 0U) {return false;}
  value.erase(0U, prefix.size());
  return true;
}

// Sets both capability flags from `source` and returns the label that remains.
inline std::string take_prior_capabilities(std::string source, PosePrior & prior)
{
  prior.may_seed_refinement = strip_prefix(source, kSeedCapabilityPrefix);
  prior.may_anchor_end_face = strip_prefix(source, kEndFaceCapabilityPrefix);
  return source;
}

// Scene discovery is gated for blocks resting on the ground: cluster_max_center_z drops a
// block hanging in the gripper at 2.5 m, in the post-refinement height filter, after the
// prior has already seeded and refined it. Measured on the carried-high captures (i176 and
// re-run for i178): raising the ceiling alone recovers all 8 frames at 0.070 m / <=2.7 deg
// from the FK prediction, where the shipped ceiling returns nothing. cluster_min_extent_z is
// deliberately left alone -- the seeded path never goes through the cluster proposal gates,
// so relaxing it buys this case nothing and only widens the scene's candidate surface.
//
// The relaxation is scoped to the call that asked for it: only the first
// `request_prior_count` priors come from the request, the rest the node adds from FK on every
// cloud, and a high gripper must not quietly raise the ceiling of an ordinary scene
// discovery. `may_seed_refinement` is the same capability the detector core requires before a
// prior may seed a hypothesis, so it marks exactly the requests that need an off-the-ground
// gate: a carried block (REFINE_GRASPED) and, since 326, one just set down on a high course.
// The trigger compares the prior's world z against a gate on height above ground -- the
// ground fit only exists once detection has run, and the two agree to the relief of the site.
inline DetectionParameters parameters_for_request(
  const DetectionParameters & base,
  const PosePriors & priors,
  std::size_t request_prior_count,
  double elevated_cluster_max_center_z)
{
  auto parameters = base;
  const std::size_t count = std::min(request_prior_count, priors.size());
  for (std::size_t index = 0; index < count; ++index) {
    const auto & prior = priors[index];
    if (!prior.may_seed_refinement || prior.weight <= 0.0 ||
      prior.position.z() <= base.cluster_max_center_z)
    {
      continue;
    }
    parameters.cluster_max_center_z =
      std::max(base.cluster_max_center_z, elevated_cluster_max_center_z);
    break;
  }
  return parameters;
}

// A prior may only seed a hypothesis, or anchor an end face, when its dimensions match the
// detector's block_dims exactly, and the two come from different config files in different
// repositories. A mismatch makes REFINE_GRASPED -- and since 331 either REFINE_BLOCK -- a
// silent no-op, so the caller reports it.
inline bool request_prior_dims_mismatch(
  const DetectionParameters & base,
  const PosePriors & priors,
  std::size_t request_prior_count)
{
  const std::size_t count = std::min(request_prior_count, priors.size());
  for (std::size_t index = 0; index < count; ++index) {
    const auto & prior = priors[index];
    if ((prior.may_seed_refinement || prior.may_anchor_end_face) && prior.weight > 0.0 &&
      prior.dims != base.block_dims)
    {
      return true;
    }
  }
  return false;
}

struct TouchingRowSplit
{
  bool enabled{false};
  // Largest seam the strip may show and still count as one row. DBSCAN already
  // separates neighbours at dbscan_eps, so anything arriving here as one box is
  // tighter than that; this only stops a second component that happens to lie in
  // the same slab from being tiled in.
  double seam_gap_m{0.12};
  // |strip - n * block length| budget, i.e. the summed joints of an n-block row.
  double extent_tolerance_m{0.12};
};

// Blocks that physically abut leave no density valley at the seam, so the core's
// split_connected_row declines and the whole strip is fitted as one region. One
// block-sized box then lands anywhere along it and jumps 0.4-1.0 m between clouds
// (lab day 11-09 16-04, two blocks at the hedge: top strip 1.87 m, x-histogram flat
// across the seam). example_wall rows are 0.92 m pitch and the storage pile is two
// blocks end to end, so every post-place refine beside a neighbour meets it.
//
// The count comes from the geometry the merged box already sits on: measure the
// strip of top-face returns in the box's own slab, on both in-plane axes because
// a merged fit is as often rotated across the row as along it, and if the strip
// tiles into n block lengths it holds n blocks. Nothing is invented -- the split
// replaces one box with n boxes over the same points, and only when every tile
// survives the same evidence gate the parent passed. The cuts run across the row,
// so each tile keeps the top plane and the long side plane that constrained the
// parent; that is why the parent's observed face count carries over.
//
// Emitting the tiling as a *competing* hypothesis instead was measured and is
// net-negative (blockpose experiments/multi_block_hypotheses.md): the evidence
// score rates a box on its own support, not on how much of the strip stays
// unexplained, so a half-covered tile scores as well as the right one. Replacing,
// gated on all n tiles clearing min_score, does not ask the score that question.
inline std::size_t split_touching_rows(
  DetectionResult & detection,
  const DetectionParameters & params,
  const TouchingRowSplit & config,
  const SensorContext * sensor_context,
  const PosePriors * priors)
{
  if (!config.enabled || detection.hypotheses.empty()) {return 0U;}
  // Same ceiling as the core's density-valley splitter.
  constexpr std::size_t kMaxBlocks = 8U;
  std::vector<CuboidHypothesis> kept, tiles;
  std::vector<std::string> replaced_trace_ids;
  for (const auto & parent : detection.hypotheses) {
    const auto & dims = parent.pose.dims;
    const double length = std::max(dims[0], dims[1]), width = std::min(dims[0], dims[1]);
    if (!(length > 0.0)) {
      kept.push_back(parent);
      continue;
    }
    // The top face in box-local z, with the same surface band refinement uses.
    const double top_local_z = dims[2] / 2.0;
    Eigen::Index axis = 0;
    double low = 0.0, high = 0.0, strip = 0.0;
    for (Eigen::Index candidate = 0; candidate < 2; ++candidate) {
      std::vector<double> along;
      for (const auto & point : detection.above_support_points) {
        const Point local =
          parent.pose.rotation.transpose() * (point - parent.pose.position);
        if (std::abs(local[1 - candidate]) <= width / 2.0 + params.refine_band &&
          std::abs(local.z() - top_local_z) <= params.refine_band)
        {
          along.push_back(local[candidate]);
        }
      }
      std::sort(along.begin(), along.end());
      if (along.size() < 3U) {continue;}
      // Grow out from the parent's own centre only: a neighbour DBSCAN did
      // separate lies past a seam wider than seam_gap_m and is not this row.
      std::size_t pivot = 0U;
      while (pivot + 1U < along.size() && along[pivot + 1U] <= 0.0) {++pivot;}
      std::size_t first = pivot, last = pivot;
      while (first > 0U && along[first] - along[first - 1U] <= config.seam_gap_m) {--first;}
      while (last + 1U < along.size() &&
        along[last + 1U] - along[last] <= config.seam_gap_m) {++last;}
      // The beam smears a few returns past each end face -- 8 cm of it on the
      // worst lab-day hedge cloud, enough to move a tile by half that. Measure
      // between the 1st and 99th percentile of the run instead of its extremes.
      const std::size_t trim = std::max<std::size_t>(1U, (last - first + 1U) / 100U);
      if (last < first + 2U * trim) {continue;}
      const double candidate_low = along[first + trim], candidate_high = along[last - trim];
      if (candidate_high - candidate_low > strip) {
        axis = candidate;
        low = candidate_low;
        high = candidate_high;
        strip = candidate_high - candidate_low;
      }
    }
    const auto count = static_cast<std::size_t>(std::llround(strip / length));
    if (count < 2U || count > kMaxBlocks ||
      std::abs(strip - static_cast<double>(count) * length) > config.extent_tolerance_m)
    {
      kept.push_back(parent);
      continue;
    }
    // A tile's long axis follows the row, whichever way the parent was fitted.
    Eigen::Matrix3d rotation;
    rotation.col(0) = parent.pose.rotation.col(axis);
    rotation.col(2) = parent.pose.rotation.col(2);
    rotation.col(1) = rotation.col(2).cross(rotation.col(0));
    std::vector<CuboidHypothesis> row;
    for (std::size_t section = 0; section < count; ++section) {
      auto tile = parent;
      tile.pose.rotation = rotation;
      tile.pose.dims = {{length, width, dims[2]}};
      tile.pose.position += rotation.col(0) *
        (low + (static_cast<double>(section) + 0.5) * strip / static_cast<double>(count));
      tile.lineage_index.reset();
      tile.source = "row_split";
      tile.trace_id = parent.trace_id + "/row-" + std::to_string(section);
      row.push_back(std::move(tile));
    }
    // Refined together, so adjacent tiles compete for the same returns through
    // assign_points instead of both sliding onto the better-observed block.
    std::vector<CuboidPose> initial;
    for (const auto & tile : row) {
      initial.push_back({tile.pose.position, tile.pose.rotation, tile.pose.dims, 0.0, "row_split"});
    }
    RefineParameters refine_params;
    refine_params.huber_scale = params.refine_huber_scale;
    refine_params.max_translation = params.refine_max_translation;
    refine_params.max_rotation_deg = params.refine_max_rotation_deg;
    refine_params.min_points = params.refine_min_points;
    const auto refined = refine_poses(
      initial, detection.above_support_points, params.refine_band, params.refine_iterations,
      refine_params);
    const auto support = assign_points(
      refined, detection.above_support_points, params.refine_band);
    bool accepted = true;
    for (std::size_t section = 0; section < row.size(); ++section) {
      auto & tile = row[section];
      tile.pose.position = refined[section].position;
      tile.pose.rotation = refined[section].rotation;
      const auto scored = make_hypothesis(
        tile.pose, support[section].size(), detection.ground,
        parent.evidence.observed_geometry_faces, sensor_context);
      tile.evidence = scored.evidence;
      tile.support_height_m = scored.support_height_m;
      tile.prior_match = best_prior_match(
        tile.pose.position, tile.pose.rotation, tile.pose.dims, priors);
      accepted = accepted && tile.evidence.score >= params.min_score;
      for (std::size_t other = 0; other < section; ++other) {
        accepted = accepted && !boxes_overlap(tile.pose, row[other].pose);
      }
    }
    if (!accepted) {
      kept.push_back(parent);
      continue;
    }
    replaced_trace_ids.push_back(parent.trace_id);
    tiles.insert(tiles.end(), row.begin(), row.end());
  }
  if (tiles.empty()) {return 0U;}
  kept.insert(kept.end(), tiles.begin(), tiles.end());
  detection.hypotheses = select_hypotheses(std::move(kept));
  std::sort(
    detection.hypotheses.begin(), detection.hypotheses.end(),
    [](const auto & a, const auto & b) {return a.pose.position.z() > b.pose.position.z();});
  detection.poses.clear();
  for (const auto & hypothesis : detection.hypotheses) {detection.poses.push_back(hypothesis.pose);}
  detection.counts.selected_hypotheses = detection.hypotheses.size();
  populate_diagnostics(detection);
  for (auto & trace : detection.refined_candidate_trace) {
    for (const auto & id : replaced_trace_ids) {
      if (trace.id == id || trace.id == id + "/refined") {trace.fate = "row_split_replaced";}
    }
  }
  return tiles.size();
}


struct RequestOptions
{
  bool refine_enabled{true};
  double elevated_cluster_max_center_z{3.0};
  TouchingRowSplit touching_row_split;
};

struct RequestResult
{
  DetectionParameters parameters;  // after parameters_for_request
  DetectionResult detection;
  std::size_t row_split_blocks{0U};
};

// One detector request on already filtered world points. `priors` holds the request's own
// priors first (`request_prior_count` of them), then the node's FK prior.
inline RequestResult detect_request(
  const Points & points, const DetectionParameters & base, const RequestOptions & options,
  const SensorContext * sensor_context, const PosePriors & priors,
  std::size_t request_prior_count, const VisualScorer * visual_scorer = nullptr)
{
  RequestResult result;
  result.parameters = parameters_for_request(
    base, priors, request_prior_count, options.elevated_cluster_max_center_z);
  result.detection = options.refine_enabled ?
    detect(points, result.parameters, sensor_context, &priors, visual_scorer) :
    detect_without_refinement(points, result.parameters, sensor_context, &priors, visual_scorer);
  result.row_split_blocks = split_touching_rows(
    result.detection, result.parameters, options.touching_row_split, sensor_context, &priors);
  return result;
}

}  // namespace blockpose::core
