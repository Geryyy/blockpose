#include <blockpose/core/proposals.hpp>

#include <gtest/gtest.h>

namespace blockpose::core { namespace {

TEST(Evidence, SupportsVisibleFaceAndRejectsThroughVolumeReturn) {SensorContext c; c.origin=Point(-2,0,0); c.ray_directions={Point(1,0,0),Point(1,.8,0)}; c.ranges={1.55,5.0}; const auto e=visibility_evidence(Point::Zero(),Eigen::Matrix3d::Identity(),{{.9,.6,.6}},c); EXPECT_EQ(e.expected_faces,1U); EXPECT_EQ(e.covered_faces,1U); EXPECT_EQ(e.supported_rays,1U); EXPECT_EQ(e.violations,0U);}
TEST(Evidence, MissRayCrossingVolumeIsConclusive) {SensorContext c;c.origin=Point(-2,0,0);c.miss_directions={Point(1,0,0)};EXPECT_EQ(free_space_violations_from_misses(Point::Zero(),Eigen::Matrix3d::Identity(),{{.9,.6,.6}},c),1U);}
TEST(Evidence, NoContextHypothesisRetainsPythonZeroRayFeatureBehavior) {Pose pose;pose.position.z()=.3;GroundPlane ground;const auto h=make_hypothesis(pose,200U,ground,1U);EXPECT_EQ(h.evidence.expected_visible_faces,0U);EXPECT_EQ(h.evidence.supported_rays,0U);EXPECT_NEAR(h.evidence.score,.25*.5+.15,1e-12);}
TEST(Evidence, ContextPopulatesAllEvidenceAndScoreFeatures) {Pose pose;pose.position.z()=.3;GroundPlane ground;SensorContext c;c.origin=Point(-2,0,.3);c.ray_directions={Point(1,0,0)};c.ranges={1.55};const auto h=make_hypothesis(pose,400U,ground,1U,&c);EXPECT_EQ(h.evidence.expected_visible_faces,1U);EXPECT_EQ(h.evidence.covered_visible_faces,1U);EXPECT_EQ(h.evidence.supported_rays,1U);EXPECT_EQ(h.evidence.incident_rays,1U);EXPECT_NEAR(h.evidence.score,1.,1e-12);}

}}  // namespace
