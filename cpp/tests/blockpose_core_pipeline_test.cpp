#include <blockpose/core/pipeline.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace blockpose::core {namespace {

// A block lying with its long axis radial to the lidar: centre (0, 0, .3), yaw 0, and
// the cloud holds only the +x end face plus 16 cm of top.  Its x extent is .16 m, under
// cluster_min_extent_xy, so no proposal component ever forms for it (326).
// `half_width` over .3 makes the same patch too wide to be a block's end face.
Points synthetic_end_face_only_block(double half_width = .3) {
  const int columns = static_cast<int>(2. * half_width / .04 + .5);
  Points points;
  for (int x = -30; x <= 30; ++x) {
    for (int y = -30; y <= 30; ++y) {points.emplace_back(x * .04, y * .04, 0.);}
  }
  for (int y = 0; y <= columns; ++y) {
    for (int z = 0; z <= 15; ++z) {points.emplace_back(.45, -half_width + y * .04, z * .04);}
  }
  for (int x = 1; x <= 4; ++x) {
    for (int y = 0; y <= columns; ++y) {
      points.emplace_back(.45 - x * .04, -half_width + y * .04, .6);
    }
  }
  return points;
}

// Two blocks end to end along x with `gap` between them: the row's two top faces and
// its two outer end faces.  Abutting blocks leave no density valley at the seam, so
// only the extent can tell the splitter that the row holds two of them.
Points synthetic_abutting_row(double gap) {
  const double half = .9 + gap / 2.;
  Points points;
  for (int x = -40; x <= 40; ++x) {
    for (int y = -20; y <= 20; ++y) {points.emplace_back(x * .04, y * .04, 0.);}
  }
  for (int y = 0; y <= 15; ++y) {
    for (int x = 0; x <= 22; ++x) {
      points.emplace_back(-half + x * .04, -.3 + y * .04, .6);
      points.emplace_back(half - x * .04, -.3 + y * .04, .6);
    }
    for (int z = 0; z <= 15; ++z) {
      points.emplace_back(-half, -.3 + y * .04, z * .04);
      points.emplace_back(half, -.3 + y * .04, z * .04);
    }
  }
  return points;
}

// Two blocks stacked, seen as the upper one's top face plus the flush side face of
// both.  The lower block's own top face is covered by the block on it, so no top
// plane and therefore no plane pairing reaches it.
Points synthetic_two_stack() {
  Points points;
  for (int x = -30; x <= 30; ++x) {
    for (int y = -30; y <= 30; ++y) {points.emplace_back(x * .04, y * .04, 0.);}
  }
  for (int x = 0; x <= 22; ++x) {
    for (int y = 0; y <= 15; ++y) {points.emplace_back(-.44 + x * .04, -.3 + y * .04, 1.2);}
  }
  for (int y = 0; y <= 15; ++y) {
    for (int z = 0; z <= 30; ++z) {points.emplace_back(.45, -.3 + y * .04, z * .04);}
  }
  return points;
}

// The Seibersdorf pile seen from -y: two blocks end to end along x with a 2 cm seam
// and one block crosswise on top of both, which covers the middle of their top faces.
// Truth: bottom (-.46, 0, .3) and (.46, 0, .3) at yaw 0, top (0, 0, .9) at yaw 90.
Points synthetic_crosswise_pile() {
  Points points;
  for (int x = -40; x <= 40; ++x) {
    for (int y = -30; y <= 30; ++y) {points.emplace_back(x * .04, y * .04, 0.);}
  }
  for (int x = 0; x <= 45; ++x) {
    const double along = -.91 + x * .04;
    for (int y = 0; y <= 15; ++y) {
      if (std::abs(along) > .3) {points.emplace_back(along, -.3 + y * .04, .6);}
    }
    for (int z = 0; z <= 15; ++z) {points.emplace_back(along, -.3, z * .04);}
  }
  for (int x = 0; x <= 15; ++x) {
    for (int y = 0; y <= 22; ++y) {points.emplace_back(-.3 + x * .04, -.45 + y * .04, 1.2);}
    for (int z = 0; z <= 15; ++z) {points.emplace_back(-.3 + x * .04, -.45, .6 + z * .04);}
  }
  return points;
}

Points synthetic_block_with_ground() {Points points; for (int x = -30; x <= 30; ++x) {for (int y = -30; y <= 30; ++y) {points.emplace_back(x * .04, y * .04, 0.);}} for (int x = 0; x <= 22; ++x) {for (int y = 0; y <= 15; ++y) {points.emplace_back(-.44 + x * .04, -.3 + y * .04, .6);}} for (int y = 0; y <= 15; ++y) {for (int z = 0; z <= 15; ++z) {points.emplace_back(.45, -.3 + y * .04, z * .04);}} return points;}

TEST(DetectorCorePipeline, VoxelOutputIsCanonicalAndInputOrderIndependent) {Points first{Point(.01,.01,.01), Point(.03,.03,.03), Point(.12,.01,.01)}; auto second = first; std::reverse(second.begin(), second.end()); const auto down_a = voxel_downsample(first, .1), down_b = voxel_downsample(second, .1); ASSERT_EQ(down_a.size(), 2U); ASSERT_EQ(down_a.size(), down_b.size()); EXPECT_NEAR((down_a[0] - down_b[0]).norm(), 0., 1e-12); EXPECT_NEAR(down_a[0].x(), .02, 1e-8);}

TEST(DetectorCorePipeline, ComposesGroundProposalPlaneAndSelectionStages) {DetectionParameters params; params.min_inliers = 30; params.ransac_iterations = 600; const auto result = detect_without_refinement(synthetic_block_with_ground(), params); EXPECT_GT(result.counts.downsampled_points, 100U); EXPECT_GT(result.counts.above_support_points, 100U); EXPECT_GE(result.counts.proposal_components, 1U); EXPECT_GT(result.counts.plane_fit_calls, 0U); EXPECT_GE(result.counts.raw_hypotheses, 1U); ASSERT_FALSE(result.poses.empty()); EXPECT_NEAR(result.poses.front().position.z(), .3, .06); ASSERT_FALSE(result.refined_candidate_trace.empty()); EXPECT_EQ(result.refined_candidate_trace.front().stage, "pre_refinement_pre_selection");}

TEST(DetectorCorePipeline, PreserveTopAxisRetainsPlaneTiltAndRefinedYaw) {const Eigen::Matrix3d initial = exp_so3(Point(.04, -.03, .20)); const Eigen::Matrix3d refined = exp_so3(Point(.20, -.15, .35)); const Eigen::Matrix3d guarded = preserve_top_axis(initial, refined); EXPECT_NEAR((guarded.col(2) - initial.col(2)).norm(), 0., 1e-12); EXPECT_NEAR(guarded.determinant(), 1., 1e-12); EXPECT_TRUE((guarded.transpose() * guarded).isApprox(Eigen::Matrix3d::Identity(), 1e-12));}

TEST(DetectorCorePipeline, PosePriorResolvesYawButCannotCreateEvidence) {Pose aligned; aligned.position = Point(1., 2., .3); Pose rotated = aligned; rotated.rotation = Eigen::AngleAxisd(M_PI / 2., Point::UnitZ()).toRotationMatrix(); GroundPlane ground; auto supported = make_hypothesis(aligned, 100U, ground, 2U); auto competing = make_hypothesis(rotated, 100U, ground, 2U); supported.evidence.score = .60; competing.evidence.score = .68; PosePrior prior; prior.source = "fk"; prior.position = aligned.position; prior.rotation = aligned.rotation; prior.weight = .20; PosePriors priors{prior}; supported.prior_match = best_prior_match(supported.pose.position, supported.pose.rotation, supported.pose.dims, &priors); competing.prior_match = best_prior_match(competing.pose.position, competing.pose.rotation, competing.pose.dims, &priors); const auto selected = select_hypotheses({competing, supported}); ASSERT_EQ(selected.size(), 1U); EXPECT_NEAR(selected.front().pose.rotation(0, 0), 1., 1e-12); EXPECT_NEAR(cuboid_orientation_error_rad(aligned.rotation, Eigen::AngleAxisd(M_PI, Point::UnitZ()).toRotationMatrix()), 0., 1e-12); EXPECT_EQ(best_prior_match(Point::Zero(), Eigen::Matrix3d::Identity(), aligned.dims, nullptr).score, 0.0);}

TEST(DetectorCorePipeline, PriorYawAnchorsEndFaceOnlyBlock) {
  const auto cloud = synthetic_end_face_only_block();
  PosePrior prior;
  prior.source = "registered_block:block_2";
  prior.may_anchor_end_face = true;  // yaw only; seeding stays off (303)
  prior.position = Point(.05, .02, .3);  // believed pose, off by 54 mm
  prior.rotation = Eigen::Matrix3d::Identity();
  prior.weight = .20;
  prior.translation_tolerance_m = .35;
  PosePriors priors{prior};
  const auto anchored = detect(cloud, {}, nullptr, &priors);
  ASSERT_EQ(anchored.hypotheses.size(), 1U);
  EXPECT_LT((anchored.hypotheses.front().pose.position - Point(0., 0., .3)).norm(), .05);
  EXPECT_EQ(anchored.hypotheses.front().evidence.observed_geometry_faces, 1U);
  // Without the yaw there is nothing to anchor to, and the capability is what grants
  // it: an otherwise identical prior that may not anchor leaves the block unseen.
  EXPECT_TRUE(detect(cloud, {}, nullptr, nullptr).hypotheses.empty());
  priors.front().may_anchor_end_face = false;
  EXPECT_TRUE(detect(cloud, {}, nullptr, &priors).hypotheses.empty());

  // A belief that leads the block by more than half its length must return nothing, not a
  // box behind the far face: the face the box is hung on is chosen by density, so the fit
  // stays at the block and the request window then rejects it.
  priors.front().may_anchor_end_face = true;
  priors.front().position = Point(.6, 0., .3);
  EXPECT_TRUE(detect(cloud, {}, nullptr, &priors).hypotheses.empty());

  // Same geometry, 2 m across: a patch that is not a block's width is not its end face.
  priors.front().position = Point(.05, .02, .3);
  EXPECT_TRUE(
    detect(synthetic_end_face_only_block(1.), {}, nullptr, &priors).hypotheses.empty());
}

TEST(DetectorCorePipeline, AbuttingRowIsSplitWithoutADensityValley) {
  for (const double gap : {0., .02}) {
    const auto result = detect(synthetic_abutting_row(gap));
    ASSERT_EQ(result.poses.size(), 2U) << "gap " << gap;
    std::vector<double> centres{result.poses[0].position.x(), result.poses[1].position.x()};
    std::sort(centres.begin(), centres.end());
    EXPECT_NEAR(centres.front(), -(.9 + gap) / 2., .05) << "gap " << gap;
    EXPECT_NEAR(centres.back(), (.9 + gap) / 2., .05) << "gap " << gap;
  }
}

TEST(DetectorCorePipeline, BottomOfAStackIsFoundFromTheSharedSideFace) {
  const auto result = detect(synthetic_two_stack());  // poses sort by descending z
  ASSERT_EQ(result.poses.size(), 2U);
  EXPECT_LT((result.poses.front().position - Point(0., 0., .9)).norm(), .08);
  EXPECT_LT((result.poses.back().position - Point(0., 0., .3)).norm(), .08);
  EXPECT_EQ(result.hypotheses.back().evidence.observed_geometry_faces, 1U);
}

// The pile's top block lies across the row under it, so the footprint of the whole
// region reads the row's axis, not the block's: before this it came out at the
// believed centre with the yaw 90 degrees out, which refinement cannot undo.
TEST(DetectorCorePipeline, CrosswiseTopOfAPileKeepsItsOwnYaw) {
  DetectionParameters params; params.cluster_max_center_z = 1.5;  // as detector.yaml
  const auto result = detect(synthetic_crosswise_pile(), params);
  ASSERT_EQ(result.poses.size(), 3U);
  const auto & top = result.poses.front();  // poses sort by descending z
  EXPECT_LT((top.position - Point(0., 0., .9)).norm(), .05);
  EXPECT_NEAR(std::abs(top.rotation(1, 0)), 1., .05) << "long axis across the row";
  for (std::size_t index = 1U; index < result.poses.size(); ++index) {
    EXPECT_NEAR(std::abs(result.poses[index].rotation(0, 0)), 1., .05) << "bottom " << index;
    EXPECT_NEAR(std::abs(result.poses[index].position.x()), .46, .06) << "bottom " << index;
  }
}

TEST(DetectorCorePipeline, FkPriorSeedsOnlyCloudSupportedRefinement) {DetectionParameters params; params.cluster_min_size = 1000U; params.refine_min_points = 20; params.refine_band = .06; PosePrior prior; prior.source = "fk"; prior.may_seed_refinement = true; prior.position = Point(0., 0., .3); prior.rotation = Eigen::Matrix3d::Identity(); prior.weight = .35; PosePriors priors{prior}; const auto seeded = detect(synthetic_block_with_ground(), params, nullptr, &priors); ASSERT_EQ(seeded.hypotheses.size(), 1U); EXPECT_EQ(seeded.hypotheses.front().prior_match.source, "fk"); EXPECT_NEAR(seeded.hypotheses.front().pose.position.z(), .3, .06); ASSERT_EQ(seeded.refined_candidate_trace.size(), 1U); const auto & trace = seeded.refined_candidate_trace.front(); EXPECT_EQ(trace.id, "seed/fk/0/refined"); EXPECT_EQ(trace.source, "fk_seed"); EXPECT_EQ(trace.stage, "post_refinement_pre_nms"); EXPECT_FALSE(trace.lineage_index.has_value()); EXPECT_EQ(trace.fate, "final"); EXPECT_EQ(trace.prior_match.source, "fk"); EXPECT_NEAR(trace.selection_score, trace.evidence.score + trace.prior_match.score + (trace.visual_evidence.available ? trace.visual_evidence.score : 0.0), 1e-12); const auto unsupported = detect(synthetic_block_with_ground(), params, nullptr, nullptr); EXPECT_TRUE(unsupported.hypotheses.empty()); EXPECT_TRUE(unsupported.refined_candidate_trace.empty());}

}}  // namespace blockpose::core
