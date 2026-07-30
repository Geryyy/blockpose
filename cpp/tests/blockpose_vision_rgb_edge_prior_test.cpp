#include <blockpose/vision/rgb_edge_prior.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <array>

namespace blockpose::vision
{
namespace
{
core::Pose pose(double yaw = 0.0)
{
  core::Pose value;
  value.position = core::Point(0.0, 0.0, 5.0);
  value.rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  value.dims = {{1.4, 0.6, 0.6}};
  return value;
}

cv::Mat wireframe(const core::Pose & value, const cv::Matx33d & camera)
{
  cv::Mat image = cv::Mat::zeros(120, 160, CV_8UC1);
  const std::array<std::array<int, 3>, 8> signs{{{{-1, -1, -1}}, {{-1, -1, 1}}, {{-1, 1, -1}}, {{-1, 1, 1}}, {{1, -1, -1}}, {{1, -1, 1}}, {{1, 1, -1}}, {{1, 1, 1}}}};
  std::array<cv::Point, 8> points;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector3d local(.5 * signs[index][0] * value.dims[0], .5 * signs[index][1] * value.dims[1], .5 * signs[index][2] * value.dims[2]);
    const Eigen::Vector3d world = value.rotation * local + value.position;
    points[index] = cv::Point(cvRound(camera(0, 0) * world.x() / world.z() + camera(0, 2)), cvRound(camera(1, 1) * world.y() / world.z() + camera(1, 2)));
  }
  for (const auto edge : std::array<std::array<int, 2>, 12>{{{{0, 1}}, {{1, 3}}, {{3, 2}}, {{2, 0}}, {{4, 5}}, {{5, 7}}, {{7, 6}}, {{6, 4}}, {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}}}) {
    cv::line(image, points[edge[0]], points[edge[1]], cv::Scalar(255));
  }
  return image;
}
}  // namespace

TEST(RgbEdgePrior, CorrectProjectionBeatsYawSibling)
{
  const cv::Matx33d camera(100.0, 0.0, 80.0, 0.0, 100.0, 60.0, 0.0, 0.0, 1.0);
  RgbEdgePriorParameters parameters;
  parameters.edge_percentile = 85.0;
  const RgbEdgePrior prior(
    wireframe(pose(), camera), camera, cv::Mat(), Eigen::Isometry3d::Identity(), cv::Mat(), parameters);
  ASSERT_TRUE(prior.ready());
  const auto correct = prior.score(pose());
  const auto wrong = prior.score(pose(M_PI / 2.0));
  ASSERT_TRUE(correct.available);
  ASSERT_TRUE(wrong.available);
  EXPECT_GT(correct.score, wrong.score);
}
}  // namespace blockpose::vision
