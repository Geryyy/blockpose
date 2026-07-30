#pragma once

// Classical, candidate-conditioned RGB evidence.  This is deliberately an
// optional tie-breaker: it scores only poses supplied by the geometric core.
#include <blockpose/core/proposals.hpp>

#include <opencv2/core.hpp>

#include <Eigen/Geometry>

namespace blockpose::vision
{
struct RgbEdgePriorParameters
{
  double weight{0.05};
  double edge_percentile{92.0};
  double sample_spacing_px{4.0};
  double distance_scale_px{3.0};
  double min_support_fraction{0.15};
  int min_samples{12};
  int max_image_dimension_px{768};
};

class RgbEdgePrior
{
public:
  RgbEdgePrior(
    const cv::Mat & grayscale, const cv::Matx33d & camera_matrix,
    const cv::Mat & distortion, const Eigen::Isometry3d & world_from_camera,
    cv::Mat excluded_mask, RgbEdgePriorParameters parameters);

  bool ready() const;
  core::VisualEvidence score(const core::Pose & pose) const;

private:
  cv::Matx33d camera_matrix_;
  cv::Mat distortion_;
  Eigen::Isometry3d world_from_camera_;
  cv::Mat excluded_mask_;
  cv::Mat distance_px_;
  cv::Mat gradient_x_;
  cv::Mat gradient_y_;
  RgbEdgePriorParameters parameters_;
  cv::Size source_size_;
};
}  // namespace blockpose::vision
