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

}}  // namespace blockpose::core
