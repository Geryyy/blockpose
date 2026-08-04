// Stand-in for the ROS node: consumes the installed package exactly as a
// downstream ament/catkin target would, and touches both changed headers.
#include "blockpose/core/pipeline.hpp"
#include "blockpose/core/proposals.hpp"
#include "blockpose/core/gripper_self_filter.hpp"
#include <cassert>
#include <cstdio>
int main() {
  blockpose::core::DetectionParameters pp;
  std::printf("cluster_max_extent_xy = %.2f\n", pp.cluster_max_extent_xy);
  assert(pp.cluster_max_extent_xy == 2.25);

  blockpose::core::Pzs100SelfFilterParameters fp;
  fp.outboard_extent_m = 0.30; fp.cross_rail_extent_m = 0.90;
  fp.rail_end_margin_m = 0.0;  fp.inner_clearance_m = 0.08;
  blockpose::core::validate_pzs100_self_filter_parameters(fp);
  blockpose::core::Pzs100RailPose rail;
  rail.parent_position = {0.0, 0.35, 2.0};
  rail.rail_position   = {0.8, 0.35, 2.0};
  rail.world_from_parent_rotation << 1,0,0, 0,0,1, 0,-1,0;
  rail.outward_axis_parent = {0.0, 0.0, 1.0};
  const auto box = blockpose::core::make_pzs100_gripper_filter_box(rail, fp);
  std::printf("jaw box thickness = %.2f (expect 0.22 = 0.30 - 0.08)\n", box.box.size.y());
  assert(std::abs(box.box.size.y() - 0.22) < 1e-9);
  std::printf("ROS-style consumer OK\n");
  return 0;
}
