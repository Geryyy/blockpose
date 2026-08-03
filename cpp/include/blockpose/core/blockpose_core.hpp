#pragma once

// Public portable scene-discovery API.  It contains no ROS, PCL, image, or
// model-runtime dependency; adapters provide points, priors, and diagnostics.
#include <blockpose/core/evidence.hpp>
#include <blockpose/core/annotation_snap.hpp>
#include <blockpose/core/geometry.hpp>
#include <blockpose/core/gripper_self_filter.hpp>
#include <blockpose/core/pipeline.hpp>
#include <blockpose/core/priors.hpp>
#include <blockpose/core/proposals.hpp>
#include <blockpose/core/refine.hpp>
