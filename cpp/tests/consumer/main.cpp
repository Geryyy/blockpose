// Stand-in for the ROS node: consumes the installed package exactly as a
// downstream ament/catkin target would, and touches both changed headers.
//
// This is a release GATE, not a smoke test: it must fail loudly (non-zero
// exit) if the shipped defaults a downstream node relies on drift. Do not
// use assert() for the checks below -- tools/check_cpp_consumer.sh and any
// real ROS/ament build configure this consumer with -DCMAKE_BUILD_TYPE=Release,
// which defines NDEBUG and compiles assert() to nothing. A version of this
// file that used assert() would print the numbers, "pass", and exit 0 no
// matter what the values were -- exactly the silent-drift failure mode this
// gate exists to catch. REQUIRE() below is our own check that cannot be
// compiled out by the consumer's build type.
#include "blockpose/core/pipeline.hpp"
#include "blockpose/core/proposals.hpp"
#include "blockpose/core/gripper_self_filter.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
void require(bool condition, const char *expr, const char *file, int line, const char *why) {
  if (condition) return;
  // Flush the human-readable printf trail first: when a CI job captures this
  // to a pipe, stdout is fully buffered and the failure message would
  // otherwise land above the value that caused it.
  std::fflush(stdout);
  std::fprintf(stderr, "CONSUMER GATE FAILED: %s (%s:%d)\n  %s\n", expr, file, line, why);
  std::exit(1);
}
}  // namespace
#define REQUIRE(cond, why) require((cond), #cond, __FILE__, __LINE__, (why))

int main() {
  blockpose::core::DetectionParameters pp;
  std::printf("cluster_max_extent_xy = %.2f\n", pp.cluster_max_extent_xy);
  REQUIRE(pp.cluster_max_extent_xy == 2.25,
          "cluster_max_extent_xy drifted from the shipped 2.25. This is the "
          "component-width gate pipeline.hpp applies before plane-fitting; at "
          "1.5 it silently cost 11 points of recall (see "
          "experiments/detector_cluster_failures.md). The Python dataclass "
          "default in blockpose/core_types.py and this C++ default must not "
          "drift apart -- tests/python/test_detector_sweep.py guards the "
          "Python side of the same contract.");
  std::printf("dbscan_eps = %.2f\n", pp.dbscan_eps);
  REQUIRE(pp.dbscan_eps == 0.07,
          "dbscan_eps drifted from the shipped 0.07. This controls the "
          "clustering radius the ROS node's detections are built on; a "
          "downstream consumer compiling against a different default would "
          "silently detect a different set of clusters than the one this "
          "package was measured against (see docs/production-readiness-plan.md).");

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
  REQUIRE(std::abs(box.box.size.y() - 0.22) < 1e-9,
          "PZS100 gripper self-filter jaw-box derivation changed: expected "
          "outboard_extent_m(0.30) - inner_clearance_m(0.08) == 0.22. A "
          "downstream ROS node uses this box to reject gripper-jaw returns "
          "from the point cloud before detection; a wrong thickness here "
          "either leaves jaw returns in (false detections) or eats real "
          "block returns near the jaws.");
  std::printf("ROS-style consumer OK\n");
  return 0;
}
