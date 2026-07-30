#include <blockpose/core/proposals.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace blockpose::core {namespace {

Points make_top_and_side() {Points points; for (int x = 0; x <= 9; ++x) {for (int y = 0; y <= 6; ++y) {points.emplace_back(-.45 + x * .1, -.3 + y * .1, .6);}} for (int y = 0; y <= 6; ++y) {for (int z = 0; z <= 6; ++z) {points.emplace_back(.45, -.3 + y * .1, z * .1);}} return points;}

TEST(DetectorCoreProposals, FitsPlanesPairsAndSynthesizes) {DetectionParameters params; params.min_inliers = 20; params.ransac_iterations = 500; const auto cloud = make_top_and_side(); const auto fitted = fit_planes(cloud, params); ASSERT_GE(fitted.first.size(), 2U); GroundPlane ground; ground.normal = Point::UnitZ(); const auto pairs = candidate_plane_sets(fitted.first, params, ground, params.block_dims); const auto found = std::find_if(pairs.begin(), pairs.end(), [](const auto & value) {return value.second != nullptr;}); ASSERT_NE(found, pairs.end()); const auto pose = synthesize_pose(*found->first, found->second, cloud, params.block_dims); EXPECT_NEAR(pose.position.z(), .3, .03); EXPECT_NEAR(std::abs(pose.rotation.col(2).dot(Point::UnitZ())), 1., 1e-6);}

TEST(DetectorCoreProposals, SatNmsKeepsTouchingAndRejectsInterpenetration) {Pose first; first.position = Point(0., 0., .3); Pose overlap = first; overlap.position.x() = .1; Pose touching = first; touching.position.x() = .9; EXPECT_TRUE(boxes_overlap(first, overlap)); EXPECT_FALSE(boxes_overlap(first, touching)); GroundPlane ground; auto a = make_hypothesis(first, 100, ground, 2); a.evidence.score = .8; auto b = make_hypothesis(overlap, 100, ground, 2); b.evidence.score = .7; auto c = make_hypothesis(touching, 100, ground, 2); c.evidence.score = .6; const auto selected = select_hypotheses({b, c, a}); ASSERT_EQ(selected.size(), 2U); EXPECT_NEAR(selected[0].pose.position.x(), 0., 1e-9);}

TEST(DetectorCoreProposals, SpatialGridHandlesBoundedDenseStressDeterministically) {Points cloud; for(int group=0;group<80;++group) for(int x=0;x<8;++x) for(int y=0;y<5;++y) cloud.emplace_back(group*.25+x*.008,y*.008,.3+(x%2)*.008); DetectionParameters params;params.dbscan_eps=.025;params.dbscan_min_points=4;params.cluster_min_size=10;params.cluster_max_size=100;params.cluster_min_extent_xy=.02;params.region_max_extent_xy=.2;params.cluster_min_extent_z=.001;params.cluster_max_extent_z=.1;params.proposal_max_components=80;const auto first=dbscan_proposals(cloud,params);auto reversed=cloud;std::reverse(reversed.begin(),reversed.end());const auto second=dbscan_proposals(reversed,params);ASSERT_EQ(first.size(),80U);ASSERT_EQ(second.size(),first.size());for(std::size_t i=0;i<first.size();++i){EXPECT_EQ(first[i].size(),40U);EXPECT_EQ(first[i],second[i]);}}

}}  // namespace blockpose::core
