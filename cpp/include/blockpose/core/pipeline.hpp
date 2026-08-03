#pragma once

// Composition of the non-refinement DetectFree stages.  ROS converts clouds
// into Points and calls this; it owns neither TF nor persistent world state.
#include "blockpose/core/proposals.hpp"
#include "blockpose/core/refine.hpp"

#include <map>
#include <set>
#include <chrono>

namespace blockpose::core
{
struct PipelineCounts
{
  std::size_t input_points{0}, downsampled_points{0}, above_support_points{0};
  std::size_t proposal_components{0}, plane_regions{0}, plane_fit_calls{0};
  std::size_t plane_search_points{0}, plane_full_points_scored{0};
  std::size_t plane_trials_evaluated{0}, plane_valid_trials{0};
  std::size_t raw_hypotheses{0}, refinement_candidates{0}, selected_hypotheses{0};
};
// Milliseconds spent in the public pipeline stages.  Kept as fixed fields so
// native consumers do not need to depend on Python dictionary conventions.
struct PipelineTimings
{
  double voxel_downsample{0.0}, ground_model{0.0}, proposals{0.0}, region_split{0.0};
  double plane_search{0.0}, plane_full_score_refit{0.0};
  double hypothesis_generation_evidence{0.0}, plane_hypotheses{0.0};
  double selection{0.0}, refinement{0.0}, total{0.0};
};
// One record for every refined candidate that reaches the final NMS input.
// This intentionally mirrors contributions already used by
// hypothesis_selection_score(), rather than introducing a parallel scoring
// implementation for diagnostics or offline ablations.
struct RefinedCandidateTrace
{
  std::string id;
  std::string source;
  std::optional<std::size_t> lineage_index;
  std::string lineage_id;
  Pose pose;
  HypothesisEvidence evidence;
  PriorMatch prior_match;
  VisualEvidence visual_evidence;
  double selection_score{0.0};
  // `pre_refinement_pre_selection` when refinement is disabled, otherwise
  // `post_refinement_pre_nms`. Consumers must not mix these stages silently.
  std::string stage{"post_refinement_pre_nms"};
  // Always overwritten before the trace leaves detect()/detect_without_refinement:
  // "final", "pre_refinement_selection_rejected", or "post_refinement_nms_rejected".
  std::string fate;
};
// Final-pose evidence in the stable public diagnostic contract. Calibration is
// deliberately unavailable until a native calibration artifact is introduced.
struct DetectionDiagnostics
{
  HypothesisEvidence evidence;
  std::array<double, 5> features{};  // support, height, coverage, free-space, surface
  std::string observability{"limited"};
  std::optional<double> proposal_scale_m;
  double support_height_m{0.0};
};
struct DetectionResult {
  std::vector<Pose> poses; std::vector<CuboidHypothesis> hypotheses; PipelineCounts counts;
  PipelineTimings timings;
  std::vector<DetectionDiagnostics> diagnostics;
  LocalGroundModel ground; Points above_support_points; std::vector<RawHypothesisLineage> raw_lineage;
  std::vector<RefinedCandidateTrace> refined_candidate_trace;
  // Diagnostic only, populated below alongside raw_lineage and consumed by
  // nothing downstream of detect_without_refinement -- see the struct
  // comments in detector_core_proposals.hpp for what each field means.
  std::vector<ProposalComponentDiagnostics> proposal_diagnostics;
  std::vector<WideProposalSkipDiagnostics> wide_proposal_skips;
  std::vector<PlaneRegionDiagnostics> plane_region_diagnostics;
  std::vector<PlanePatchDiagnostics> plane_patch_diagnostics;
  std::vector<RegionPairingDiagnostics> pairing_diagnostics;
};

inline DetectionDiagnostics diagnostics_for(const CuboidHypothesis & hypothesis)
{
  DetectionDiagnostics out;
  out.evidence = hypothesis.evidence;
  const auto & e = out.evidence;
  const double total_rays = static_cast<double>(e.supported_rays + e.free_space_violations);
  out.features = {{
    std::min(static_cast<double>(e.support_points) / 400.0, 1.0),
    std::max(0.0, 1.0 - e.top_height_error_m / 0.2),
    e.expected_visible_faces ? static_cast<double>(e.covered_visible_faces) / e.expected_visible_faces : 0.0,
    total_rays ? static_cast<double>(e.supported_rays) / total_rays : 0.0,
    e.incident_rays ? static_cast<double>(e.supported_rays) / e.incident_rays : 0.0,
  }};
  const std::size_t observed_faces = std::max(e.covered_visible_faces, e.observed_geometry_faces);
  const double clean = out.features[3];
  out.observability = observed_faces >= 3U && clean >= .98 ? "full" :
    (observed_faces >= 2U && clean >= .90 ? "partial" : "limited");
  out.proposal_scale_m = hypothesis.proposal_scale_m;
  out.support_height_m = hypothesis.support_height_m;
  return out;
}

inline void populate_diagnostics(DetectionResult & result)
{
  result.diagnostics.clear();
  result.diagnostics.reserve(result.hypotheses.size());
  for (const auto & hypothesis : result.hypotheses) {
    result.diagnostics.push_back(diagnostics_for(hypothesis));
  }
}

// Average points in floor-indexed voxels. Canonical ordering makes this
// independent of input ordering and preserves stable downstream processing.
inline Points voxel_downsample(const Points & input, double voxel_size)
{
  if (!std::isfinite(voxel_size) || voxel_size <= 0.0) {throw std::invalid_argument("voxel_size must be positive");}
  struct Accumulator {Point sum{Point::Zero()}; std::size_t count{0U};};
  std::map<std::tuple<long long, long long, long long>, Accumulator> voxels;
  for (const auto & input_point : canonical_order(input)) {
    if (!input_point.allFinite()) {throw std::invalid_argument("points must be finite");}
    const auto index = std::make_tuple(
      static_cast<long long>(std::floor(input_point.x() / voxel_size)),
      static_cast<long long>(std::floor(input_point.y() / voxel_size)),
      static_cast<long long>(std::floor(input_point.z() / voxel_size)));
    auto & value = voxels[index]; value.sum += input_point; ++value.count;
  }
  Points output;
  output.reserve(voxels.size());
  for (const auto & [_, value] : voxels) {
    output.emplace_back(value.sum / static_cast<double>(value.count));
  }
  return canonical_order(std::move(output));
}

inline std::vector<Points> split_connected_row(const Points & cluster, const DetectionParameters & params)
{
  const auto e = extent(cluster); const std::size_t axis = e[0] >= e[1] ? 0U : 1U;
  const double width = params.block_dims[axis]; const int count = static_cast<int>(std::llround(e[axis] / width));
  if (count < 2 || count > 8 || std::abs(e[axis] - count * width) > .12) {return {cluster};}
  double low = std::numeric_limits<double>::infinity(); for (const auto & point : cluster) {low = std::min(low, point[static_cast<Eigen::Index>(axis)]);}
  const double valley_half = std::max(params.voxel_size * 1.5, .04), shoulder = std::max(params.voxel_size * 3., .08);
  for (int section = 1; section < count; ++section) {
    const double boundary = low + section * width; std::size_t valley = 0, left = 0, right = 0;
    for (const auto & point : cluster) {const double value = point[static_cast<Eigen::Index>(axis)]; if (std::abs(value - boundary) <= valley_half) {++valley;} if (value >= boundary - shoulder - valley_half && value < boundary - valley_half) {++left;} if (value > boundary + valley_half && value <= boundary + shoulder + valley_half) {++right;}}
    const double expected = static_cast<double>(left + right) * valley_half / shoulder;
    if (expected <= 0.0 || valley > .45 * expected) {return {cluster};}
  }
  std::vector<Points> result; for (int section = 0; section < count; ++section) {const double begin = low + section * width, end = low + (section + 1) * width; Points region; for (const auto & point : cluster) {const double value = point[static_cast<Eigen::Index>(axis)]; if (value >= begin - 1e-6 && (value <= end + (section == count - 1 ? 1e-6 : 0.0))) {region.push_back(point);}} if (region.size() >= params.cluster_min_size) {result.push_back(std::move(region));}}
  return result.size() == static_cast<std::size_t>(count) ? result : std::vector<Points>{cluster};
}

// Builds the candidate trace from the hypotheses that fed a selection stage.
// `refined_suffix` selects between the pre-refinement id/stage convention
// (detect_without_refinement's own candidates) and the post-refinement one
// (detect()'s rescored candidates, id suffixed "/refined").
inline void populate_candidate_trace(
  DetectionResult & result, const std::vector<CuboidHypothesis> & candidates,
  const std::string & stage, bool refined_suffix)
{
  result.refined_candidate_trace.clear();
  result.refined_candidate_trace.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    RefinedCandidateTrace trace;
    trace.id = refined_suffix ? candidate.trace_id + "/refined" : candidate.trace_id;
    trace.source = candidate.source;
    trace.lineage_index = candidate.lineage_index;
    if (candidate.lineage_index && *candidate.lineage_index < result.raw_lineage.size()) {
      trace.lineage_id = result.raw_lineage[*candidate.lineage_index].id;
    }
    trace.pose = candidate.pose;
    trace.evidence = candidate.evidence;
    trace.prior_match = candidate.prior_match;
    trace.visual_evidence = candidate.visual_evidence;
    trace.selection_score = hypothesis_selection_score(candidate);
    trace.stage = stage;
    result.refined_candidate_trace.push_back(std::move(trace));
  }
}

// Marks each trace entry "final" iff its id survived into result.hypotheses,
// else `rejected_fate`. Mirrors the id convention `populate_candidate_trace`
// used to build the corresponding trace.
inline void mark_candidate_trace_fate(
  DetectionResult & result, const std::string & rejected_fate, bool refined_suffix)
{
  std::set<std::string> selected_trace_ids;
  for (const auto & hypothesis : result.hypotheses) {
    selected_trace_ids.insert(refined_suffix ?
      (hypothesis.trace_id.empty() ? "" : hypothesis.trace_id + "/refined") : hypothesis.trace_id);
  }
  for (auto & trace : result.refined_candidate_trace) {
    trace.fate = selected_trace_ids.count(trace.id) > 0U ? "final" : rejected_fate;
  }
}

inline DetectionResult detect_without_refinement(
  const Points & input, const DetectionParameters & params = {},
  const SensorContext * sensor_context = nullptr, const PosePriors * priors = nullptr,
  const VisualScorer * visual_scorer = nullptr,
  // Non-null only when a caller (detect()) plans to immediately replace the
  // pre-refinement trace once refinement proceeds. Skipping the build here
  // and handing back the exact candidates that would have fed it lets that
  // caller build the trace itself only on the path that still needs it (see
  // detect()'s early return), instead of building it on every call just to
  // discard it.
  std::vector<CuboidHypothesis> * thresholded_out = nullptr)
{
  const auto started = std::chrono::steady_clock::now();
  DetectionResult result; result.counts.input_points = input.size();
  const auto finish = [&]() {
      populate_diagnostics(result);
      result.timings.total = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
      return result;
    };
  if (input.size() < 3U) {return finish();}
  auto stage_started = std::chrono::steady_clock::now();
  const Points downsampled = voxel_downsample(input, params.voxel_size); result.timings.voxel_downsample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_started).count(); result.counts.downsampled_points = downsampled.size();
  if (downsampled.size() < 3U) {return finish();}
  GroundParameters ground_params;
  ground_params.thickness = params.ground_thickness;
  ground_params.ransac_distance = params.ground_ransac_distance;
  ground_params.ransac_iterations = params.ground_ransac_iterations;
  ground_params.ransac_seed = params.ransac_seed;
  ground_params.normal_min_z = params.ground_normal_min_z;
  ground_params.local_cell_size = params.local_ground_cell_size;
  ground_params.local_clearance = params.local_ground_clearance;
  stage_started = std::chrono::steady_clock::now();
  const Points coarse_ground = voxel_downsample(downsampled, .1);
  const GroundRemovalResult removed = remove_ground(downsampled, ground_params, &coarse_ground); result.timings.ground_model = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_started).count(); result.ground = removed.ground; result.above_support_points = removed.above_ground; result.counts.above_support_points = removed.above_ground.size();
  if (removed.above_ground.empty()) {return finish();}
  // Keep the low-clearance cloud for cuboid refinement, but do not let
  // ground-adjacent returns seed DBSCAN components. Mixing the two causes
  // cluttered blocks to choose a different side plane.
  Points proposal_seed;
  proposal_seed.reserve(removed.above_ground.size());
  for (const auto & point : removed.above_ground) {
    if (result.ground.height(point) >= params.ground_thickness) {
      proposal_seed.push_back(point);
    }
  }
  stage_started = std::chrono::steady_clock::now();
  const auto proposals = dbscan_proposals(
    proposal_seed, params, &result.ground, &result.proposal_diagnostics, sensor_context);
  result.timings.proposals = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_started).count();
  result.counts.proposal_components = proposals.size();
  stage_started = std::chrono::steady_clock::now();
  std::vector<CuboidHypothesis> raw;
  for (std::size_t proposal_index = 0; proposal_index < proposals.size(); ++proposal_index) {
    const auto & proposal = proposals[proposal_index];
    const DetectionParameters local_params = density_scaled_parameters(
      params, proposal, sensor_context);
    const auto split_started = std::chrono::steady_clock::now();
    const auto regions = split_connected_row(proposal, local_params);
    result.timings.region_split += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - split_started).count();
    const auto proposal_extent = extent(proposal);
    // Match _detect_blocks_impl: a connected region wider than a single block
    // is only admissible when the density-valley splitter supplied observable
    // boundaries.  Do not tile a seamless long object into known cuboids.
    if (!params.fit_wide_proposals && regions.size() == 1U &&
      (proposal_extent[0] > params.cluster_max_extent_xy ||
      proposal_extent[1] > params.cluster_max_extent_xy))
    {
      // Diagnostic only: this component never reaches fit_planes below, so
      // it would otherwise leave zero trace in any other diagnostic here.
      WideProposalSkipDiagnostics skip; skip.proposal_component = proposal_index; skip.point_count = proposal.size(); skip.region_count = regions.size(); skip.extent = proposal_extent;
      result.wide_proposal_skips.push_back(std::move(skip));
      continue;
    }
    for (std::size_t region_index = 0; region_index < regions.size(); ++region_index) {
      const auto & region = regions[region_index];
      ++result.counts.plane_regions; const auto [planes, counts] = fit_planes(region, local_params);
      result.counts.plane_fit_calls += counts.calls; result.counts.plane_search_points += counts.search_points; result.counts.plane_full_points_scored += counts.full_points_scored; result.counts.plane_trials_evaluated += counts.trials_evaluated; result.counts.plane_valid_trials += counts.valid_trials; result.timings.plane_search += counts.search_ms; result.timings.plane_full_score_refit += counts.full_score_refit_ms;
      // Diagnostic only, these two blocks: fit_planes/the patches it returned
      // are unchanged above; this just records what they were for the stages
      // before hypothesis synthesis (RawHypothesisLineage starts too late to
      // see a component/region that never produced a raw hypothesis at all).
      {PlaneRegionDiagnostics region_diag; region_diag.proposal_component = proposal_index; region_diag.region = region_index; region_diag.plane_count = planes.size(); region_diag.leftover_points = counts.leftover_points; region_diag.stop_reason = counts.stop_reason; result.plane_region_diagnostics.push_back(std::move(region_diag));}
      for (std::size_t plane_index = 0; plane_index < planes.size(); ++plane_index) {const auto & patch = planes[plane_index]; PlanePatchDiagnostics patch_diag; patch_diag.proposal_component = proposal_index; patch_diag.region = region_index; patch_diag.plane_index = plane_index; patch_diag.point_count = patch.points.size(); patch_diag.normal = patch.normal; patch_diag.centroid = patch.centroid; patch_diag.residual_mad = patch.residual_mad; result.plane_patch_diagnostics.push_back(std::move(patch_diag));}
      const auto hypothesis_started = std::chrono::steady_clock::now();
      const auto dims_values = candidate_dims(params);
      for (std::size_t dims_index = 0; dims_index < dims_values.size(); ++dims_index) {const auto & dims = dims_values[dims_index];
      RegionPairingDiagnostics region_pairing; region_pairing.proposal_component = proposal_index; region_pairing.region = region_index; region_pairing.dims_index = dims_index;
      const auto pairs = candidate_plane_sets(planes, params, result.ground, dims, &region_pairing.planes);
      result.pairing_diagnostics.push_back(std::move(region_pairing));
      for (const auto & pair : pairs) {
        Points local_support = region;
        if (pair.second != nullptr) {local_support = pair.first->points; local_support.insert(local_support.end(), pair.second->points.begin(), pair.second->points.end());}
        const auto pose = canonicalize_pose(synthesize_pose(*pair.first, pair.second, local_support, dims)); const std::size_t support = pair.first->points.size() + (pair.second == nullptr ? 0U : pair.second->points.size()); auto hypothesis = make_hypothesis(pose, support, result.ground, pair.second == nullptr ? 1U : 2U, sensor_context); hypothesis.prior_match = best_prior_match(pose.position, pose.rotation, pose.dims, priors);
        const double top_height = result.ground.height(pair.first->centroid); const bool supported = !(top_height <= dims[2] * 1.1 && std::abs(hypothesis.support_height_m) > .135);
        const std::size_t top_index = static_cast<std::size_t>(pair.first - planes.data());
        const std::optional<std::size_t> side_index = pair.second == nullptr ? std::nullopt : std::optional<std::size_t>(static_cast<std::size_t>(pair.second - planes.data()));
        RawHypothesisLineage lineage;
        lineage.id = "component-" + std::to_string(proposal_index) + "/region-" + std::to_string(region_index) + "/dims-" + std::to_string(dims_index) + "/top-" + std::to_string(top_index) + "/side-" + (side_index ? std::to_string(*side_index) : "none");
        lineage.proposal_component = proposal_index; lineage.region = region_index; lineage.top_plane = top_index; lineage.side_plane = side_index;
        lineage.top_normal = pair.first->normal; lineage.top_centroid = pair.first->centroid;
        if (pair.second != nullptr) {lineage.side_normal = pair.second->normal; lineage.side_centroid = pair.second->centroid;}
        lineage.candidate_dims = dims; lineage.top_only = pair.second == nullptr; lineage.synthesized_pose = pose; lineage.evidence = hypothesis.evidence; lineage.top_support_height_m = top_height;
        lineage.top_patch_inliers = pair.first->points.size();
        patch_in_plane_extents(
          *pair.first, pose.rotation.col(0), pose.rotation.col(1),
          &lineage.top_patch_extent_major_m, &lineage.top_patch_extent_minor_m,
          &lineage.top_patch_extent_local_x_m, &lineage.top_patch_extent_local_y_m);
        lineage.accepted_to_raw = top_height >= dims[2] * .70 && supported;
        lineage.fate = lineage.accepted_to_raw ? "raw" : "geometric_rejected";
        result.raw_lineage.push_back(std::move(lineage));
        if (result.raw_lineage.back().accepted_to_raw) {
          hypothesis.lineage_index = result.raw_lineage.size() - 1U;
          hypothesis.trace_id = result.raw_lineage.back().id;
          hypothesis.source = "plane_fit";
          raw.push_back(std::move(hypothesis));
        }
      }}
      result.timings.hypothesis_generation_evidence += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hypothesis_started).count();
    }
  }
  result.timings.plane_hypotheses = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_started).count();
  result.counts.raw_hypotheses = raw.size(); stage_started = std::chrono::steady_clock::now(); std::vector<CuboidHypothesis> thresholded; for (const auto & hypothesis : raw) {if (hypothesis.evidence.score >= params.min_score) {thresholded.push_back(hypothesis); if (hypothesis.lineage_index) {auto & lineage = result.raw_lineage[*hypothesis.lineage_index]; lineage.passed_score_threshold = true; lineage.fate = "score_passed";}} else if (hypothesis.lineage_index) {result.raw_lineage[*hypothesis.lineage_index].fate = "score_rejected";}}
  if (visual_scorer != nullptr) {for (auto & hypothesis : thresholded) {hypothesis.visual_evidence = (*visual_scorer)(hypothesis.pose);}}
  // Refinement is optional in an ablation. Preserve its exact alternative-
  // selection input as a separately labelled trace instead of pretending the
  // post-refinement contract exists when it does not -- unless the caller
  // asked to defer that (see thresholded_out above).
  if (thresholded_out != nullptr) {*thresholded_out = thresholded;} else {populate_candidate_trace(result, thresholded, "pre_refinement_pre_selection", false);}
  result.hypotheses = select_conflict_alternatives(std::move(thresholded), params.conflict_alternatives);
  for (const auto & hypothesis : result.hypotheses) {if (hypothesis.lineage_index) {result.raw_lineage[*hypothesis.lineage_index].fate = "pre_refinement_selected";}}
  for (auto & lineage : result.raw_lineage) {if (lineage.fate == "score_passed") {lineage.fate = "pre_refinement_selection_rejected";}}
  result.counts.refinement_candidates = result.hypotheses.size();
  result.hypotheses.erase(std::remove_if(result.hypotheses.begin(), result.hypotheses.end(), [&result, &params](const auto & hypothesis) {const bool rejected = result.ground.height(hypothesis.pose.position) < 0.0 || result.ground.height(hypothesis.pose.position) > params.cluster_max_center_z; if (rejected && hypothesis.lineage_index) {result.raw_lineage[*hypothesis.lineage_index].fate = "pre_refinement_bounds_rejected";} return rejected;}), result.hypotheses.end());
  if (thresholded_out == nullptr) {mark_candidate_trace_fate(result, "pre_refinement_selection_rejected", false);}
  std::sort(result.hypotheses.begin(), result.hypotheses.end(), [](const auto & a, const auto & b) {return a.pose.position.z() > b.pose.position.z();}); for (const auto & hypothesis : result.hypotheses) {result.poses.push_back(hypothesis.pose);} result.counts.selected_hypotheses = result.hypotheses.size(); result.timings.selection = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_started).count(); return finish();
}

inline Eigen::Matrix3d preserve_top_axis(
  const Eigen::Matrix3d & initial_rotation, const Eigen::Matrix3d & refined_rotation)
{
  const Point top_axis = initial_rotation.col(2).normalized();
  Point x_axis = refined_rotation.col(0) - top_axis * top_axis.dot(refined_rotation.col(0));
  if (x_axis.norm() < 1.0e-8) {
    x_axis = initial_rotation.col(0) - top_axis * top_axis.dot(initial_rotation.col(0));
  }
  x_axis.normalize();
  Eigen::Matrix3d result;
  result.col(0) = x_axis;
  result.col(1) = top_axis.cross(x_axis).normalized();
  result.col(2) = top_axis;
  return result;
}

inline DetectionResult detect(
  const Points & input, const DetectionParameters & params = {},
  const SensorContext * sensor_context = nullptr, const PosePriors * priors = nullptr,
  const VisualScorer * visual_scorer = nullptr)
{
  const auto started = std::chrono::steady_clock::now();
  // Deferred pre-refinement trace input (see detect_without_refinement's
  // thresholded_out): built into a trace only on the early-return path below,
  // never discarded on the (common) path that reaches refinement.
  std::vector<CuboidHypothesis> thresholded_before_refinement;
  DetectionResult result = detect_without_refinement(
    input, params, sensor_context, priors, visual_scorer, &thresholded_before_refinement);
  // A prior is allowed to seed refinement only when the current cloud verifies
  // it with nearby cuboid-surface returns. This recovers gripped/occluded
  // blocks that never formed a DBSCAN proposal without manufacturing a block.
  if (priors != nullptr) {
    for (std::size_t prior_index = 0; prior_index < priors->size(); ++prior_index) {
      const auto & prior = (*priors)[prior_index];
      if (prior.source != "fk" || prior.weight <= 0.0 || prior.dims != params.block_dims) {continue;}
      CuboidPose seed; seed.position = prior.position; seed.rotation = prior.rotation; seed.dims = prior.dims;
      // Refine the FK initialization against only the points its surface owns.
      // Its tolerance is deliberately the configured FK tolerance, rather than
      // the tighter generic refinement guard: grasp offsets can exceed the
      // latter, but a seed is retained only after the normal evidence gate.
      Points support_points = assign_points({seed}, result.above_support_points, params.refine_band).front();
      if (support_points.size() < params.refine_min_points) {continue;}
      RefineParameters seed_refine_params;
      seed_refine_params.huber_scale = params.refine_huber_scale;
      seed_refine_params.max_translation = prior.translation_tolerance_m;
      seed_refine_params.max_rotation_deg = std::min(
        params.refine_max_rotation_deg, prior.orientation_tolerance_rad * 180.0 / M_PI);
      seed_refine_params.min_points = params.refine_min_points;
      const auto seed_refined = refine_pose(seed, support_points, seed_refine_params).pose;
      if (params.refine_preserve_top_axis_if_gravity_worsens) {
        const Point up = result.ground.normal.normalized();
        if (std::abs(seed_refined.rotation.col(2).dot(up)) + 1.0e-6 <
          std::abs(seed.rotation.col(2).dot(up)))
        {
          seed.rotation = preserve_top_axis(seed.rotation, seed_refined.rotation);
        } else {
          seed.rotation = seed_refined.rotation;
        }
      } else {
        seed.rotation = seed_refined.rotation;
      }
      seed.position = seed_refined.position;
      support_points = assign_points({seed}, result.above_support_points, params.refine_band).front();
      if (support_points.size() < params.refine_min_points) {continue;}
      Pose seed_pose; seed_pose.position = seed.position; seed_pose.rotation = seed.rotation; seed_pose.dims = seed.dims;
      Pose pose; pose.position = seed.position; pose.rotation = seed.rotation; pose.dims = seed.dims;
      auto hypothesis = make_hypothesis(
        pose, support_points.size(), result.ground, 0U, sensor_context);
      if (hypothesis.evidence.score < params.min_score) {continue;}
      // A seed is recovery for an absent/weak candidate, not a second
      // competitor that can displace a normal geometry-derived pose merely
      // through its FK bonus. An overlapping seed must improve geometric
      // evidence by a configured margin before it can replace that candidate.
      double best_overlapping_evidence = -std::numeric_limits<double>::infinity();
      for (const auto & existing : result.hypotheses) {
        if (boxes_overlap(seed_pose, existing.pose)) {
          best_overlapping_evidence = std::max(
            best_overlapping_evidence, existing.evidence.score);
        }
      }
      if (hypothesis.evidence.score < best_overlapping_evidence +
        params.fk_seed_min_evidence_gain)
      {
        continue;
      }
      hypothesis.prior_match = best_prior_match(pose.position, pose.rotation, pose.dims, priors);
      if (visual_scorer != nullptr) {hypothesis.visual_evidence = (*visual_scorer)(pose);}
      hypothesis.trace_id = "seed/" + prior.source + "/" + std::to_string(prior_index);
      hypothesis.source = prior.source + "_seed";
      result.hypotheses.push_back(std::move(hypothesis));
    }
  }
  if (result.hypotheses.empty()) {
    // Refinement never ran. result.hypotheses only ever grows between
    // detect_without_refinement's return and this check (seeding only
    // appends), so it being empty here means it was already empty there --
    // build exactly the trace detect_without_refinement would have.
    populate_candidate_trace(result, thresholded_before_refinement, "pre_refinement_pre_selection", false);
    mark_candidate_trace_fate(result, "pre_refinement_selection_rejected", false);
    result.timings.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); return result;
  }
  std::vector<CuboidPose> initial;
  initial.reserve(result.hypotheses.size());
  for (const auto & hypothesis : result.hypotheses) {
    initial.push_back({hypothesis.pose.position, hypothesis.pose.rotation, hypothesis.pose.dims,
      hypothesis.pose.confidence, "plane_fit"});
  }
  RefineParameters refine_params;
  refine_params.huber_scale = params.refine_huber_scale;
  refine_params.max_translation = params.refine_max_translation;
  refine_params.max_rotation_deg = params.refine_max_rotation_deg;
  refine_params.min_points = params.refine_min_points;
  const auto refinement_started = std::chrono::steady_clock::now();
  const auto refined = refine_poses(
    initial, result.above_support_points, params.refine_band,
    params.refine_iterations, refine_params);
  result.timings.refinement = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - refinement_started).count();
  std::vector<CuboidHypothesis> rescored;
  rescored.reserve(result.hypotheses.size());
  for (std::size_t index = 0; index < result.hypotheses.size(); ++index) {
    Pose pose = result.hypotheses[index].pose;
    pose.position = refined[index].position;
    pose.rotation = refined[index].rotation;
    if (params.refine_preserve_top_axis_if_gravity_worsens) {
      const Point up = result.ground.normal.normalized();
      const double initial_alignment = std::abs(initial[index].rotation.col(2).dot(up));
      const double refined_alignment = std::abs(pose.rotation.col(2).dot(up));
      if (refined_alignment + 1.0e-6 < initial_alignment) {
        pose.rotation = preserve_top_axis(initial[index].rotation, pose.rotation);
      }
    }
    pose.dims = refined[index].dims;
    const auto & prior = result.hypotheses[index];
    auto hypothesis = make_hypothesis(
      pose, prior.evidence.support_points, result.ground,
      prior.evidence.observed_geometry_faces, sensor_context);
    hypothesis.lineage_index = prior.lineage_index;
    hypothesis.trace_id = prior.trace_id;
    hypothesis.source = prior.source;
    hypothesis.prior_match = best_prior_match(pose.position, pose.rotation, pose.dims, priors);
    if (visual_scorer != nullptr) {hypothesis.visual_evidence = (*visual_scorer)(pose);}
    if (result.ground.height(pose.position) >= 0.0 &&
      result.ground.height(pose.position) <= params.cluster_max_center_z)
    {
      rescored.push_back(std::move(hypothesis));
    }
  }
  for (const auto & hypothesis : rescored) {if (hypothesis.lineage_index) {result.raw_lineage[*hypothesis.lineage_index].fate = "post_refinement_candidate";}}
  populate_candidate_trace(result, rescored, "post_refinement_pre_nms", true);
  result.hypotheses = select_hypotheses(std::move(rescored));
  mark_candidate_trace_fate(result, "post_refinement_nms_rejected", true);
  for (auto & lineage : result.raw_lineage) {if (lineage.fate == "post_refinement_candidate") {lineage.fate = "post_refinement_nms_rejected";}}
  for (const auto & hypothesis : result.hypotheses) {if (hypothesis.lineage_index) {result.raw_lineage[*hypothesis.lineage_index].fate = "final";}}
  std::sort(result.hypotheses.begin(), result.hypotheses.end(), [](const auto & a, const auto & b) {
    return a.pose.position.z() > b.pose.position.z();
  });
  result.poses.clear();
  for (const auto & hypothesis : result.hypotheses) {result.poses.push_back(hypothesis.pose);}
  result.counts.selected_hypotheses = result.hypotheses.size(); populate_diagnostics(result); result.timings.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  return result;
}
}  // namespace blockpose::core
