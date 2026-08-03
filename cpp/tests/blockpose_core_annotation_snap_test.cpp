#include <blockpose/core/annotation_snap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace blockpose::core { namespace {

using Face = std::pair<int, int>;

Points cuboid_surfaces(
  const Point & center, const std::array<double, 3> & dims, double yaw_deg,
  std::initializer_list<Face> faces)
{
  const double yaw = yaw_deg * M_PI / 180.0;
  Eigen::Matrix3d rotation;
  rotation << std::cos(yaw), -std::sin(yaw), 0.0,
              std::sin(yaw), std::cos(yaw), 0.0,
              0.0, 0.0, 1.0;
  const Point half(dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0);
  Points points;
  for (const auto [axis, sign] : faces) {
    const int first = (axis + 1) % 3;
    const int second = (axis + 2) % 3;
    for (int i = 0; i < 27; ++i) {
      for (int j = 0; j < 27; ++j) {
        Point local = Point::Zero();
        local[axis] = sign * half[axis];
        local[first] = -half[first] + 2.0 * half[first] * i / 26.0;
        local[second] = -half[second] + 2.0 * half[second] * j / 26.0;
        points.push_back(rotation * local + center);
      }
    }
  }
  return points;
}

TEST(AnnotationSnap, UprightFitIsPermutationDeterministic)
{
  const Point center(1.7, 0.5, 1.05);
  const Points points = cuboid_surfaces(center, {{.6, .6, .9}}, 12.0, {{2, 1}, {0, -1}, {1, -1}});
  const Point seed = points.front();
  const auto first = annotation_snap(points, seed);
  Points reversed = points;
  std::reverse(reversed.begin(), reversed.end());
  const auto second = annotation_snap(reversed, seed);
  ASSERT_TRUE(first.success) << first.reason;
  ASSERT_TRUE(second.success) << second.reason;
  EXPECT_TRUE(first.pose.position.isApprox(second.pose.position, 0.0));
  EXPECT_TRUE(first.pose.rotation.isApprox(second.pose.rotation, 0.0));
  EXPECT_GT(std::abs(first.pose.rotation(2, 0)), .99);
}

TEST(AnnotationSnap, FitsPartialTopAndFront)
{
  const Point center(1.2, -.3, .7);
  const Points points = cuboid_surfaces(center, {{.9, .6, .6}}, 33.0, {{2, 1}, {0, -1}});
  const auto result = annotation_snap(points, points.front());
  ASSERT_TRUE(result.success) << result.reason;
  EXPECT_LT((result.pose.position - center).norm(), .04);
}

TEST(AnnotationSnap, TopOnlyReportsMissingVerticalSupport)
{
  const Points points = cuboid_surfaces(Point(1.0, 0.0, .6), {{.9, .6, .6}}, 15.0, {{2, 1}});
  const auto result = annotation_snap(points, points.front());
  EXPECT_FALSE(result.success);
  EXPECT_NE(result.reason.find("vertical"), std::string::npos);
}

TEST(AnnotationSnap, NearbyBlocksDoNotProduceACompromisePose)
{
  const Point first_center(0.0, 0.0, .6);
  Points points = cuboid_surfaces(first_center, {{.9, .6, .6}}, 0.0, {{2, 1}, {0, -1}, {1, -1}});
  const Points second = cuboid_surfaces(Point(.9, 0.0, .6), {{.9, .6, .6}}, 0.0, {{2, 1}, {0, -1}, {1, -1}});
  points.insert(points.end(), second.begin(), second.end());
  const auto result = annotation_snap(points, points.front());
  if (result.success) {
    EXPECT_LT((result.pose.position - first_center).norm(), .04);
  } else {
    EXPECT_NE(result.reason.find("ambiguous"), std::string::npos);
  }
}

TEST(AnnotationSnap, DensePlaneComponentsUseLocalNeighborSearch)
{
  Points points;
  for (int x = 0; x < 100; ++x) {
    for (int y = 0; y < 100; ++y) {points.emplace_back(.01 * x, .01 * y, 0.0);}
  }
  const auto started = std::chrono::steady_clock::now();
  const auto components = annotation_components(points, .018);
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ASSERT_EQ(components.size(), 1U);
  EXPECT_EQ(components.front().size(), points.size());
  EXPECT_LT(elapsed, 1.0);
}

}}  // namespace blockpose::core
