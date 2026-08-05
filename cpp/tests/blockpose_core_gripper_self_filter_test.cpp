#include <blockpose/core/gripper_self_filter.hpp>

#include <gtest/gtest.h>

namespace blockpose::core { namespace {

TEST(GripperSelfFilter, RemovesPointsInsideOrientedBox) {OrientedBox left_box; left_box.center = Point(-0.5, 0.0, 0.5); left_box.size = Point(1.0, 1.0, 1.0); const Points input{Point(-0.5, 0.0, 0.5), Point(-1.0, 0.0, 0.5), Point(0.01, 0.0, 0.5), Point(-0.5, 0.51, 0.5)}; std::size_t removed = 0U; const auto retained = remove_points_inside_oriented_boxes(input, {left_box}, &removed); EXPECT_EQ(removed, 2U); ASSERT_EQ(retained.size(), 2U); EXPECT_NEAR(retained[0].x(), 0.01, 1e-12); EXPECT_NEAR(retained[1].y(), 0.51, 1e-12);}

TEST(GripperSelfFilter, RejectsInvalidBox)
{
  EXPECT_THROW(
    remove_points_inside_oriented_boxes(
      Points{}, {OrientedBox{Point::Zero(), Eigen::Matrix3d::Identity(), Point::Zero()}}),
    std::invalid_argument);
  EXPECT_THROW(
    remove_points_inside_oriented_boxes(
      Points{}, {OrientedBox{Point::Zero(), 2.0 * Eigen::Matrix3d::Identity(), Point::Ones()}}),
    std::invalid_argument);
}

TEST(GripperSelfFilter, KeepsInclusiveAxisAlignedWorkspace)
{
  std::size_t removed = 0U;
  const auto kept = keep_points_inside_axis_aligned_box(
    {Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0), Point(1.1, 0.0, 0.0)},
    Point::Zero(), Point::Ones(), &removed);
  EXPECT_EQ(removed, 1U); ASSERT_EQ(kept.size(), 2U);
  EXPECT_THROW(keep_points_inside_axis_aligned_box({}, Point::Ones(), Point::Zero()), std::invalid_argument);
}

TEST(GripperSelfFilter, BuildsPzs100OutboardBoxFromRailEndpoints)
{
  Pzs100RailPose rail;
  rail.parent_position = Point(1.0, 2.0, 3.0);
  rail.rail_position = Point(1.0, 2.0, 4.0);
  rail.outward_axis_parent = Point::UnitX();
  const Pzs100SelfFilterParameters parameters{0.4, 0.6, 0.1};

  const auto result = make_pzs100_gripper_filter_box(rail, parameters);

  EXPECT_TRUE(result.rail_start.isApprox(rail.parent_position));
  EXPECT_TRUE(result.rail_end.isApprox(rail.rail_position));
  EXPECT_TRUE(result.outward_normal.isApprox(Point::UnitX()));
  EXPECT_TRUE(result.box.center.isApprox(Point(1.2, 2.0, 3.5)));
  EXPECT_TRUE(result.box.size.isApprox(Point(1.2, 0.4, 0.6)));
  EXPECT_TRUE(result.box.rotation.col(0).isApprox(Point::UnitZ()));
  EXPECT_TRUE(result.box.rotation.col(1).isApprox(Point::UnitX()));
  EXPECT_TRUE(result.box.rotation.col(2).isApprox(Point::UnitY()));
}

TEST(GripperSelfFilter, RejectsDegeneratePzs100Rail)
{
  Pzs100RailPose rail;
  EXPECT_THROW(make_pzs100_gripper_filter_box(rail), std::invalid_argument);
  rail.rail_position = Point::UnitX();
  rail.outward_axis_parent = Point::UnitX();
  EXPECT_THROW(make_pzs100_gripper_filter_box(rail), std::invalid_argument);
}

namespace {
constexpr double kDegToRad = 0.017453292519943295;

// Two level, parallel rails 2 m apart (parent->rail along +X, outward
// along +/-Y), matching a symmetric PZS100 rig with the boom directly
// above the gap between them.
std::vector<Pzs100RailPose> LevelRailPair()
{
  Pzs100RailPose left;
  left.parent_position = Point(0.0, 0.0, 0.0);
  left.rail_position = Point(1.0, 0.0, 0.0);
  left.outward_axis_parent = Point::UnitY();
  Pzs100RailPose right;
  right.parent_position = Point(0.0, -2.0, 0.0);
  right.rail_position = Point(1.0, -2.0, 0.0);
  right.outward_axis_parent = -Point::UnitY();
  return {left, right};
}
}  // namespace

TEST(GripperSelfFilter, AboveGripperExclusionSpansGapAboveBothJawsAndKeepsPayloadBelow)
{
  const auto rails = LevelRailPair();
  const Pzs100SelfFilterParameters parameters{0.3, 0.4, 0.0, 1.0, 0.1};

  const auto above = make_pzs100_above_gripper_exclusion_box(rails, parameters);

  // Top of each jaw box is at z = cross_rail_extent_m / 2 = 0.2; the volume
  // extrudes above_exclusion_height_m (1.0) upward from there, so it is
  // centred at z = 0.2 + 0.5 = 0.7.
  //
  // The horizontal footprint is the COMBINED extent of the two jaw boxes plus
  // the margin -- see the derivation comment on
  // make_pzs100_above_gripper_exclusion_box ("the combined horizontal extent
  // of both jaw boxes along each axis, so the volume covers whatever the
  // gripper actually occupies rather than a reconstruction from rail
  // geometry").  Transverse: the jaws straddle a 2 m gap and each reaches
  // outward_extent_m (0.3) beyond its own centreline, so 2.0 + 0.3 + 0.3 plus
  // the 0.1 m margin = 2.7.  Along the rails: parent->rail length 1.0 plus the
  // margin = 1.1.
  //
  // This expectation read (1.0, 2.1, 1.0) until now -- the rail-geometry
  // reconstruction that predates the footprint change.  It was never updated
  // because GTest was not installable in the dev image, so this whole suite
  // was unrunnable; see docs/ENVIRONMENT.md's C++ build prerequisites.
  EXPECT_TRUE(above.center.isApprox(Point(0.5, -1.0, 0.7), 1e-9));
  EXPECT_TRUE(above.size.isApprox(Point(1.1, 2.7, 1.0), 1e-9));
  EXPECT_TRUE(above.rotation.col(2).isApprox(Point::UnitZ(), 1e-9));

  // A point squarely above the gap between the jaws must be inside the
  // volume (and so gets removed by the filter)...
  EXPECT_TRUE(point_inside_oriented_box(Point(0.5, -1.0, 0.5), above));
  // ...but the payload block hangs BELOW the jaws: points there -- however
  // far below -- must never be treated as inside the above-gripper volume.
  // This is the safety-critical property of the whole feature.
  EXPECT_FALSE(point_inside_oriented_box(Point(0.5, -1.0, -0.5), above));
  EXPECT_FALSE(point_inside_oriented_box(Point(0.5, -1.0, -50.0), above));

  const auto rail_boxes = make_pzs100_gripper_filter_boxes(rails, parameters);
  const std::vector<OrientedBox> boxes{rail_boxes[0].box, rail_boxes[1].box, above};
  std::size_t removed = 0U;
  const auto kept = remove_points_inside_oriented_boxes(
    {Point(0.5, -1.0, 0.5), Point(0.5, -1.0, -0.5)}, boxes, &removed);
  EXPECT_EQ(removed, 1U);
  ASSERT_EQ(kept.size(), 1U);
  EXPECT_TRUE(kept[0].isApprox(Point(0.5, -1.0, -0.5)));
}

TEST(GripperSelfFilter, AboveGripperExclusionInertAtZeroHeight)
{
  const auto rails = LevelRailPair();
  const Pzs100SelfFilterParameters parameters{};  // above_exclusion_height_m defaults to 0.0

  // A zero (default, "feature off") height must never be built into a
  // volume -- the filtering entry point is expected to skip calling this
  // builder entirely in that case, and the builder itself refuses to
  // construct a degenerate zero-height box.
  EXPECT_THROW(make_pzs100_above_gripper_exclusion_box(rails, parameters), std::invalid_argument);
  // The pre-existing per-rail boxes must be completely unaffected.
  const auto rail_boxes = make_pzs100_gripper_filter_boxes(rails, parameters);
  EXPECT_EQ(rail_boxes.size(), 2U);
}

TEST(GripperSelfFilter, AboveGripperExclusionResolvesUpOnATiltedGripper)
{
  const double angle = 20.0 * kDegToRad;
  Eigen::Matrix3d tilt;
  tilt << 1.0, 0.0, 0.0,
    0.0, std::cos(angle), -std::sin(angle),
    0.0, std::sin(angle), std::cos(angle);
  const Point up_expected = tilt * Point::UnitZ();

  Pzs100RailPose left;
  left.parent_position = tilt * Point(0.0, 0.0, 0.0);
  left.rail_position = tilt * Point(1.0, 0.0, 0.0);
  left.world_from_parent_rotation = tilt;
  left.outward_axis_parent = Point::UnitY();
  Pzs100RailPose right;
  right.parent_position = tilt * Point(0.0, -2.0, 0.0);
  right.rail_position = tilt * Point(1.0, -2.0, 0.0);
  right.world_from_parent_rotation = tilt;
  right.outward_axis_parent = -Point::UnitY();

  const Pzs100SelfFilterParameters parameters{0.3, 0.4, 0.0, 1.0, 0.1};
  const auto above = make_pzs100_above_gripper_exclusion_box({left, right}, parameters);

  // "Up" is world +Z -- gravity -- NOT the tilted rig's own vertical.  This
  // assertion is inverted from what it originally checked, deliberately: the
  // structural-axis convention it used to encode is what made the old
  // 60-degree guard reject an ordinary upright gripper, so the axis was
  // re-anchored to gravity.  The equivalent Python test documents the same
  // contract ("`Up` is world +Z rather than the rig's own tilted up",
  // tests/python/test_pzs100_annotation_filter.py) and has been green
  // throughout; only this C++ suite went stale, because it was unrunnable.
  EXPECT_TRUE(above.rotation.col(2).isApprox(Point::UnitZ(), 1e-9));
  // `tilt` is still what places the rails; it just no longer decides "up".
  (void)up_expected;

  // Same safety property as the level case, but expressed in WORLD vertical,
  // because world vertical is now what decides which side of the jaws a point
  // is on.  The payload hangs below the gripper and must survive however the
  // rig is tilted; the boom above it must be excluded.
  EXPECT_FALSE(point_inside_oriented_box(Point(0.5, -1.0, -0.5), above));
  EXPECT_FALSE(point_inside_oriented_box(Point(0.5, -1.0, 0.0), above));
  EXPECT_TRUE(point_inside_oriented_box(Point(0.5, -1.0, 0.7), above));
}

namespace {
// Builds a two-rail pair tilted by `angle` about the shared rail axis (world
// X), mirroring the construction used by AboveGripperExclusionResolvesUpOnATiltedGripper
// above, so behaviour can be checked at specific tilts.  The 60-degree
// (|dot| >= 0.5) resolution limit these angles were originally chosen around
// no longer exists; they are kept because they still bracket the interesting
// range, including past vertical.
std::vector<Pzs100RailPose> TiltedRailPair(double angle)
{
  Eigen::Matrix3d tilt;
  tilt << 1.0, 0.0, 0.0,
    0.0, std::cos(angle), -std::sin(angle),
    0.0, std::sin(angle), std::cos(angle);

  Pzs100RailPose left;
  left.parent_position = tilt * Point(0.0, 0.0, 0.0);
  left.rail_position = tilt * Point(1.0, 0.0, 0.0);
  left.world_from_parent_rotation = tilt;
  left.outward_axis_parent = Point::UnitY();
  Pzs100RailPose right;
  right.parent_position = tilt * Point(0.0, -2.0, 0.0);
  right.rail_position = tilt * Point(1.0, -2.0, 0.0);
  right.world_from_parent_rotation = tilt;
  right.outward_axis_parent = -Point::UnitY();
  return {left, right};
}
}  // namespace

// Replaces AboveGripperExclusionAcceptsUpToSixtyDegreesTiltAndRefusesBeyondIt,
// whose entire premise -- a 60-degree tilt limit, beyond which building the
// volume must throw -- was deliberately removed.  That guard existed to stop a
// structural "up" axis from resolving backwards and driving the exclusion
// volume down through the payload; once "up" was re-anchored to gravity there
// is no sign to resolve, so there is nothing to guard and the limit only ever
// rejected legitimate grippers.  The old test asserted EXPECT_THROW at 61 and
// 90.01 degrees; both now build correctly, which is the intended behaviour.
TEST(GripperSelfFilter, AboveGripperExclusionResolvesUpAtAnyTiltAndRefusesOnlyStackedRails)
{
  const Pzs100SelfFilterParameters parameters{0.3, 0.4, 0.0, 1.0, 0.1};

  // Tilt no longer bounds the volume: "up" is gravity at every angle, well
  // past the 60 degrees the removed guard refused, and past vertical too.
  for (const double degrees : {20.0, 59.0, 60.0, 61.0, 89.0, 90.01}) {
    const auto rails = TiltedRailPair(degrees * kDegToRad);
    ASSERT_NO_THROW(make_pzs100_above_gripper_exclusion_box(rails, parameters))
      << "tilt " << degrees << " deg";
    const auto above = make_pzs100_above_gripper_exclusion_box(rails, parameters);
    EXPECT_TRUE(above.rotation.col(2).isApprox(Point::UnitZ(), 1e-9)) << "tilt " << degrees
                                                                     << " deg";
  }

  // The one genuinely degenerate tilt, and it fails for a different reason
  // than the old guard did: at exactly 90 degrees the two rail centrelines
  // stack vertically, leaving no horizontal gap for the volume to span.  This
  // is a statement about the geometry being unrepresentable, not about the
  // rig being too tilted to trust.
  EXPECT_THROW(
    make_pzs100_above_gripper_exclusion_box(TiltedRailPair(90.0 * kDegToRad), parameters),
    std::invalid_argument);
}

TEST(GripperSelfFilter, AboveGripperExclusionThrowsOnDegenerateOrDisagreeingRails)
{
  // Cross-rail axis exactly horizontal: cannot resolve which way is up.
  Pzs100RailPose horizontal_rail;
  horizontal_rail.parent_position = Point(0.0, 0.0, 0.0);
  horizontal_rail.rail_position = Point(0.0, 0.0, 1.0);  // rail direction = +Z
  horizontal_rail.outward_axis_parent = Point::UnitX();  // col(2) = Z x X = Y (horizontal)
  EXPECT_THROW(
    make_pzs100_above_gripper_exclusion_box(
      {horizontal_rail, horizontal_rail}, Pzs100SelfFilterParameters{0.3, 0.4, 0.0, 1.0, 0.0}),
    std::invalid_argument);

  // Two individually well-formed rails whose independently-resolved "up"
  // directions disagree: both have a shallow (10 deg) positive elevation
  // from horizontal, but mirrored across the rail axis, so after each is
  // independently sign-resolved toward +Z they still point in substantially
  // different directions.  A real rigid gripper cannot produce this, so it
  // must be rejected as degenerate rather than silently averaged.
  const double theta = 80.0 * kDegToRad;
  const Point up0(std::sin(theta), 0.0, std::cos(theta));
  const Point up1(-std::sin(theta), 0.0, std::cos(theta));
  ASSERT_LT(up0.dot(up1), 0.0);

  Pzs100RailPose left;
  left.parent_position = Point(0.0, 0.0, 0.0);
  left.rail_position = Point(0.0, 1.0, 0.0);  // rail direction = +Y
  left.outward_axis_parent = Point(-up0.z(), 0.0, up0.x());  // Y x b = up0
  Pzs100RailPose right;
  right.parent_position = Point(0.0, 0.0, -2.0);
  right.rail_position = Point(0.0, 1.0, -2.0);  // rail direction = +Y
  right.outward_axis_parent = Point(-up1.z(), 0.0, up1.x());  // Y x b = up1

  EXPECT_THROW(
    make_pzs100_above_gripper_exclusion_box(
      {left, right}, Pzs100SelfFilterParameters{0.3, 0.4, 0.0, 1.0, 0.0}),
    std::invalid_argument);
}

TEST(GripperSelfFilter, AboveGripperExclusionRequiresExactlyTwoRails)
{
  const auto rails = LevelRailPair();
  const Pzs100SelfFilterParameters parameters{0.3, 0.4, 0.0, 1.0, 0.0};
  EXPECT_THROW(
    make_pzs100_above_gripper_exclusion_box({rails[0]}, parameters), std::invalid_argument);
  EXPECT_THROW(
    make_pzs100_above_gripper_exclusion_box({rails[0], rails[1], rails[0]}, parameters),
    std::invalid_argument);
}

}}  // namespace blockpose::core
