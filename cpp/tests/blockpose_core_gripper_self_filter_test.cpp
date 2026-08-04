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
  // centred at z = 0.2 + 0.5 = 0.7.  Width spans the 2 m rail-to-rail gap
  // plus the 0.1 m margin; length matches the jaw boxes' rail length.
  EXPECT_TRUE(above.center.isApprox(Point(0.5, -1.0, 0.7), 1e-9));
  EXPECT_TRUE(above.size.isApprox(Point(1.0, 2.1, 1.0), 1e-9));
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

  // "Up" must follow the tilted rig's true vertical, not world +Z.
  EXPECT_TRUE(above.rotation.col(2).isApprox(up_expected, 1e-9));

  // Same safety property as the level case, expressed in the tilted rig's
  // own frame and then rotated into world coordinates: below the jaws
  // survives, above the jaws is excluded.
  const Point payload_world = tilt * Point(0.5, -1.0, -0.5);
  const Point above_world = tilt * Point(0.5, -1.0, 0.5);
  EXPECT_FALSE(point_inside_oriented_box(payload_world, above));
  EXPECT_TRUE(point_inside_oriented_box(above_world, above));
}

namespace {
// Builds a two-rail pair tilted by `angle` about the shared rail axis (world
// X), mirroring the construction used by AboveGripperExclusionResolvesUpOnATiltedGripper
// above, so the resolved "up" direction can be checked at specific tilts
// relative to the 60-degree (|dot| >= 0.5) resolution limit.
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

TEST(GripperSelfFilter, AboveGripperExclusionAcceptsUpToSixtyDegreesTiltAndRefusesBeyondIt)
{
  const Pzs100SelfFilterParameters parameters{0.3, 0.4, 0.0, 1.0, 0.1};

  // Just below the 60-degree limit: must resolve "up" correctly, matching
  // the tilted rig's true vertical, exactly like the 20-degree case above.
  {
    const double angle = 59.0 * kDegToRad;
    const auto rails = TiltedRailPair(angle);
    Eigen::Matrix3d tilt;
    tilt << 1.0, 0.0, 0.0,
      0.0, std::cos(angle), -std::sin(angle),
      0.0, std::sin(angle), std::cos(angle);
    const Point up_expected = tilt * Point::UnitZ();
    const auto above = make_pzs100_above_gripper_exclusion_box(rails, parameters);
    EXPECT_TRUE(above.rotation.col(2).isApprox(up_expected, 1e-9));
  }

  // Exactly at the 60-degree limit (|dot| == 0.5): the boundary is
  // inclusive, so this must still resolve rather than throw.
  {
    const double angle = 60.0 * kDegToRad;
    const auto rails = TiltedRailPair(angle);
    EXPECT_NO_THROW(make_pzs100_above_gripper_exclusion_box(rails, parameters));
  }

  // Just beyond the 60-degree limit: per the reproduced defect, letting the
  // dot-product sign check run here would resolve "up" backwards and drive
  // the exclusion volume through the payload instead of the boom. The fix
  // must refuse to build the volume at all instead of inverting it.
  {
    const double angle = 61.0 * kDegToRad;
    const auto rails = TiltedRailPair(angle);
    EXPECT_THROW(make_pzs100_above_gripper_exclusion_box(rails, parameters), std::invalid_argument);
  }

  // Reproduction values from the field defect report: tilt past 90 degrees
  // must never silently invert onto the payload -- it must throw.
  {
    const double angle = 90.01 * kDegToRad;
    const auto rails = TiltedRailPair(angle);
    EXPECT_THROW(make_pzs100_above_gripper_exclusion_box(rails, parameters), std::invalid_argument);
  }
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
