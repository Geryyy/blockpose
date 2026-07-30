#include <blockpose/core/gripper_self_filter.hpp>
#include <blockpose/core/pipeline.hpp>

#include <cassert>

int main()
{
  using namespace blockpose::core;

  const Points input{
    Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0),
    Point(0.0, 0.0, 0.7), Point(0.9, 0.0, 0.7), Point(0.0, 0.6, 0.7)};
  const auto ground = remove_ground(input, GroundParameters{});
  assert(ground.ground.normal.allFinite());

  OrientedBox exclusion;
  exclusion.center = Point(0.0, 0.0, 0.0);
  exclusion.size = Point(0.1, 0.1, 0.1);
  std::size_t removed = 0U;
  const auto filtered = remove_points_inside_oriented_boxes(input, {exclusion}, &removed);
  assert(removed == 1U);
  assert(filtered.size() + removed == input.size());

  DetectionParameters params;
  params.voxel_size = 0.05;
  const DetectionResult result = detect(filtered, params);
  assert(result.counts.input_points == filtered.size());
  return 0;
}
