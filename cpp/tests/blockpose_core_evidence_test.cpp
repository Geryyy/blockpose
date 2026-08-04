#include <blockpose/core/pipeline.hpp>

#include <gtest/gtest.h>

namespace blockpose::core { namespace {

TEST(Evidence, SupportsVisibleFaceAndRejectsThroughVolumeReturn) {SensorContext c; c.origin=Point(-2,0,0); c.ray_directions={Point(1,0,0),Point(1,.8,0)}; c.ranges={1.55,5.0}; const auto e=visibility_evidence(Point::Zero(),Eigen::Matrix3d::Identity(),{{.9,.6,.6}},c); EXPECT_EQ(e.expected_faces,1U); EXPECT_EQ(e.covered_faces,1U); EXPECT_EQ(e.supported_rays,1U); EXPECT_EQ(e.violations,0U);}
TEST(Evidence, MissRayCrossingVolumeIsConclusive) {SensorContext c;c.origin=Point(-2,0,0);c.miss_directions={Point(1,0,0)};EXPECT_EQ(free_space_violations_from_misses(Point::Zero(),Eigen::Matrix3d::Identity(),{{.9,.6,.6}},c),1U);}
TEST(Evidence, NoContextHypothesisRetainsPythonZeroRayFeatureBehavior) {Pose pose;pose.position.z()=.3;GroundPlane ground;const auto h=make_hypothesis(pose,200U,ground,1U);EXPECT_EQ(h.evidence.expected_visible_faces,0U);EXPECT_EQ(h.evidence.supported_rays,0U);EXPECT_NEAR(h.evidence.score,.25*.5+.15,1e-12);}
TEST(Evidence, ContextPopulatesAllEvidenceAndScoreFeatures) {Pose pose;pose.position.z()=.3;GroundPlane ground;SensorContext c;c.origin=Point(-2,0,.3);c.ray_directions={Point(1,0,0)};c.ranges={1.55};const auto h=make_hypothesis(pose,400U,ground,1U,&c);EXPECT_EQ(h.evidence.expected_visible_faces,1U);EXPECT_EQ(h.evidence.covered_visible_faces,1U);EXPECT_EQ(h.evidence.supported_rays,1U);EXPECT_EQ(h.evidence.incident_rays,1U);EXPECT_NEAR(h.evidence.score,1.,1e-12);}
TEST(Evidence, NativeDiagnosticsMatchObservabilityContract) {CuboidHypothesis hypothesis; hypothesis.evidence.support_points=200U; hypothesis.evidence.top_height_error_m=.02; hypothesis.evidence.expected_visible_faces=3U; hypothesis.evidence.covered_visible_faces=3U; hypothesis.evidence.observed_geometry_faces=2U; hypothesis.evidence.supported_rays=49U; hypothesis.evidence.free_space_violations=1U; hypothesis.evidence.incident_rays=70U; hypothesis.support_height_m=.02; const auto diagnostic=diagnostics_for(hypothesis); EXPECT_EQ(diagnostic.observability,"full"); EXPECT_NEAR(diagnostic.features[0],.5,1e-12); EXPECT_NEAR(diagnostic.features[1],.9,1e-12); EXPECT_NEAR(diagnostic.features[3],.98,1e-12); EXPECT_NEAR(diagnostic.features[4],.7,1e-12);}
TEST(Evidence, NativeDiagnosticsRemainAlignedWithFinalHypotheses) {DetectionResult result; result.hypotheses.resize(2U); result.hypotheses[0].evidence.observed_geometry_faces=1U; result.hypotheses[1].evidence.observed_geometry_faces=2U; result.hypotheses[1].evidence.supported_rays=9U; result.hypotheses[1].evidence.free_space_violations=1U; populate_diagnostics(result); ASSERT_EQ(result.diagnostics.size(),2U); EXPECT_EQ(result.diagnostics[0].observability,"limited"); EXPECT_EQ(result.diagnostics[1].observability,"partial");}

// A known occluder (the robot's own gripper) must not turn into evidence
// against a hypothesis standing behind it. Measured on real captures, heavily
// occluded blocks are 76% surface-hidden by the gripper, and every ray it
// stops was previously counted as incident-but-unsupported -- penalising the
// hypothesis twice, through covered/expected and supported/incident.
TEST(Evidence, KnownOccluderNeitherSupportsNorRefutesAHypothesisBehindIt) {
  SensorContext context;
  context.origin = Point(-2, 0, 0);
  context.ray_directions = {Point(1, 0, 0)};
  context.ranges = {0.5};  // the ray stops on the occluder at x = -1.5
  const auto without = visibility_evidence(
    Point::Zero(), Eigen::Matrix3d::Identity(), {{.9, .6, .6}}, context);
  // Without occluder knowledge the ray looks like a return that failed to land
  // on the block: expected but not covered, incident but not supported.
  EXPECT_EQ(without.expected_faces, 1U);
  EXPECT_EQ(without.covered_faces, 0U);
  EXPECT_EQ(without.incident_rays, 1U);
  EXPECT_EQ(without.supported_rays, 0U);

  OrientedBox gripper;
  gripper.center = Point(-1.5, 0, 0);
  gripper.rotation = Eigen::Matrix3d::Identity();
  gripper.size = Point(0.2, 1.0, 1.0);
  context.occluders = {gripper};
  const auto with = visibility_evidence(
    Point::Zero(), Eigen::Matrix3d::Identity(), {{.9, .6, .6}}, context);
  // With it, the ray is simply uninformative: it drops out entirely rather
  // than counting against the hypothesis.
  EXPECT_EQ(with.expected_faces, 0U);
  EXPECT_EQ(with.covered_faces, 0U);
  EXPECT_EQ(with.incident_rays, 0U);
  EXPECT_EQ(with.supported_rays, 0U);
}

TEST(Evidence, AnUnoccludedReturnIsUnaffectedByDeclaringADistantOccluder) {
  SensorContext context;
  context.origin = Point(-2, 0, 0);
  context.ray_directions = {Point(1, 0, 0)};
  context.ranges = {1.55};  // lands on the block face at x = -0.45
  OrientedBox elsewhere;
  elsewhere.center = Point(0, 5, 0);
  elsewhere.rotation = Eigen::Matrix3d::Identity();
  elsewhere.size = Point(1, 1, 1);
  context.occluders = {elsewhere};
  const auto e = visibility_evidence(
    Point::Zero(), Eigen::Matrix3d::Identity(), {{.9, .6, .6}}, context);
  EXPECT_EQ(e.expected_faces, 1U);
  EXPECT_EQ(e.covered_faces, 1U);
  EXPECT_EQ(e.supported_rays, 1U);
}

}}  // namespace
