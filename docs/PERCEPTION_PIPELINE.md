# Concrete-Block Perception Pipeline — Reference

Purpose: hand this to an agent building a **new detector**. It documents the current
perception pipeline as it actually runs: node graph, one-shot modes, and the complete
ROS 2 interface surface (topics, services, actions, message definitions).

> Scope note: this describes the **real / rosbag** perception path
> (`concrete_block_perception` + `concrete_block_world_model` + `ros2_yolos_cpp`).
> Where a detector plugs in is called out explicitly in
> [§8 Building a new detector](#8-building-a-new-detector).

---

## 1. High-level architecture

The pipeline is **on-demand / one-shot**. Nodes sit idle and do work only when the
orchestrator receives a `RunPoseEstimation` request (typically from a behavior tree).

```
                 run_pose_estimation (srv)
 BT / caller ───────────────────────────────►  world_model_node
                                                (PerceptionOrchestratorNode)
                                                      │
   image  ─(synced)─►  cache                          │  orchestrates per request:
   points ─(synced)─►  cache                          │
                                                      ├─►  yolos_segmentor_service /segment   (detect + mask)
                                                      ├─►  /extract_mask_cutout               (mask ∩ cloud → cutout)
                                                      └─►  register_block (action)            (ICP pose)
                                                      │
                                                      ▼
                                       persistent world model (BlockArray)
                                       + RViz markers + planning scene
```

Segmentation → cutout → registration is driven **directly** by the orchestrator, which
also does detection→block association internally. The standalone
`block_detection_tracking_node` (service `~/track`) is launched but is **not** called by
the one-shot orchestrator path — treat it as an optional/legacy building block.

### Runtime nodes (from `perception.launch.py`)

| Node (runtime name) | Executable / package | Role |
|---|---|---|
| `yolos_segmentor_service` | `ros2_yolos_cpp` (included `segmentor_service.launch.py`) | YOLO instance segmentation as a **service** |
| `block_detection_tracking_node` | `concrete_block_perception` | Detection tracker service `~/track` (optional; not in one-shot path) |
| `registration_node` | `block_registration_node` / `concrete_block_perception` | `register_block` action + cutout service (ICP against CAD template) |
| `seyond_points_cloudini_decoder` | `cloudini_topic_converter` / `cloudini_ros` | Decompress `/seyond/points/cloudini` → `/seyond/points` |
| `world_model_node` | `world_model_node` / `concrete_block_world_model` | **PerceptionOrchestratorNode** — the brain; persistent world model |

> Note: the C++ class constructs the node as `block_world_model_node`, but the launch file
> overrides the name to **`world_model_node`**, so all its services live under
> `/world_model_node/...`.

---

## 2. One-shot modes

Modes are selected by the `mode` string in `RunPoseEstimation.srv`
(parsed in `world_model_utils.cpp`). Valid values:

| Mode | `target_block_id` | What it does |
|---|---|---|
| `SCENE_DISCOVERY` | ignored | Detect + register **all** visible blocks once; associate to existing world-model blocks or create new ones. Optional coarse fallback creates a `POSE_COARSE` block from mask+cloud centroid when registration fails. |
| `REFINE_BLOCK` | required (e.g. `wm_block_1`) | Re-register **one known** block and replace its pose. Optional `refine_block.use_pose_roi=true` seeds an ROI from the persisted pose. |
| `REFINE_GRASPED` | optional | Refine the **currently grasped** block. If empty, target = newest block with `task_status == TASK_MOVE`. Uses FK+ROI and fuses registration translation with FK orientation. |

Anything else → `kNone` (rejected).

### Per-block pose/task status enums (`Block.msg`)

- `pose_status`: `POSE_UNKNOWN=0`, `POSE_COARSE=1`, `POSE_PRECISE=2`
- `task_status`: `TASK_UNKNOWN=0`, `TASK_FREE=1`, `TASK_MOVE=2`, `TASK_PLACED=3`, `TASK_REMOVED=4`
- `goal_status`: `GOAL_NONE=0`, `GOAL_SET=1`

### ⚠️ Dead legacy

Continuous/streaming perception has been **removed**. The `perception_mode`
(`IDLE`/`CONTINUOUS`) launch arg, `SetPerceptionMode.srv`, the `pipeline_mode` arg, and all
`.../timing/continuous_*` topics are inert scaffolding. Ignore them for new work.

---

## 3. Topics

Names below are the **remapped** (fully-qualified) names from `perception.launch.py`.

### Inputs (consumed by `world_model_node`, time-synced)

| Topic | Type | Notes |
|---|---|---|
| `/blackfly_rotated/image_rect` → `image` | `sensor_msgs/Image` | Rectified RGB, synced with cloud |
| `/seyond/points` → `points` | `sensor_msgs/PointCloud2` | Lidar cloud (decoded from cloudini, ~10 Hz) |
| `/seyond/points/cloudini` | `point_cloud_interfaces/CompressedPointCloud2` | Compressed source, decoded by cloudini node |
| camera info topic (param `refine_grasped.camera_info_topic`) | `sensor_msgs/CameraInfo` | For FK+ROI projection |

### Outputs

| Topic | Type | Notes |
|---|---|---|
| `/cbp/block_world_model` | `concrete_block_world_model_interfaces/BlockArray` | Persistent world model (QoS depth 10) |
| `/cbp/block_world_model_markers` | `visualization_msgs/MarkerArray` | RViz, transient_local |
| `/cbp/block_goal_markers` | `visualization_msgs/MarkerArray` | Assembly goal poses, transient_local |

### Debug (transient_local, only when the matching debug flag is on)

| Topic | Type |
|---|---|
| `/cbp/debug/detection_overlay` | `sensor_msgs/Image` |
| `/cbp/debug/yolo_service_debug_image` | `sensor_msgs/Image` |
| `/cbp/debug/refine_grasped_roi_input` | `sensor_msgs/Image` |
| `/cbp/debug/registration/{mask_cutout,cleaned_cutout,plane_cloud,template,mask,diagnostics,gripper_boxes}` | mixed (`Image`/`PointCloud2`/`MarkerArray`) — from `registration_node` |

### Segmentor service node also publishes

`~/timing`, `~/timing/{total_ms,convert_ms,infer_ms,pack_ms,segmentations}` (`std_msgs/Float64*`) — profiling only.

---

## 4. Services

### Provided by `world_model_node` (namespaced `/world_model_node/...`)

| Service | Type | Purpose |
|---|---|---|
| `~/run_pose_estimation` | `concrete_block_world_model_interfaces/RunPoseEstimation` | **Main entry point** — trigger a one-shot mode |
| `~/get_coarse_blocks` | `.../GetCoarseBlocks` | Query cached coarse blocks (freshness-aware) |
| `~/get_planning_scene` | `.../GetPlanningScene` | Blocks + static obstacles as planning-scene objects |
| `~/set_block_task_status` | `.../SetBlockTaskStatus` | Set task status; optionally capture grasp offset for FK tracking |
| `~/upsert_block` | `.../UpsertBlock` | Add/update a block (used by sim bridge / external) |
| `~/set_block_goal` | `.../SetBlockGoal` | Assign assembly goal pose to a block |
| `~/clear_block_goals` | `.../ClearBlockGoals` | Clear goals (all or by id) |
| `~/clear_world_model` | `.../ClearWorldModel` | Wipe all blocks (keeps static objects) |

### Provided by the processing stack

| Service | Type | Node |
|---|---|---|
| `/yolos_segmentor_service/segment` | `ros2_yolos_cpp/SegmentImage` | `yolos_segmentor_service` |
| `/extract_mask_cutout` | `concrete_block_perception_interfaces/ExtractMaskCutout` | `registration_node` |
| `register_block_pose` | `concrete_block_perception_interfaces/RegisterBlock` | `registration_node` (sync alt to the action) |
| `~/track` | `concrete_block_perception_interfaces/TrackDetections` | `block_detection_tracking_node` (optional) |

### Consumed by the orchestrator (clients)

`/yolos_segmentor_service/segment`, `/extract_mask_cutout`, and the `register_block` action.

---

## 5. Actions

| Action | Type | Server |
|---|---|---|
| `register_block` | `concrete_block_perception_interfaces/RegisterBlock` | `registration_node` |

---

## 6. Interface definitions (verbatim)

### 6.1 Segmentation — `ros2_yolos_cpp/srv/SegmentImage`

```
# Request
sensor_msgs/Image image
float32 conf_threshold
float32 nms_threshold
bool return_debug
---
# Response
vision_msgs/Detection2DArray detections
sensor_msgs/Image mask
sensor_msgs/Image debug_image
bool success
string message
```

### 6.2 Registration action — `concrete_block_perception_interfaces/action/RegisterBlock`

```
# Goal
sensor_msgs/Image mask
sensor_msgs/PointCloud2 cloud
string object_class
bool use_pose_prior                  # optional world-frame pose prior
geometry_msgs/Pose prior_pose
float32 prior_position_sigma_m
float32 prior_orientation_sigma_rad
---
# Result
bool success
geometry_msgs/Pose pose
float32 fitness
float32 rmse
---
# Feedback
string stage
float32 progress
float32 elapsed_ms
```

### 6.3 Registration service — `concrete_block_perception_interfaces/srv/RegisterBlock`

```
sensor_msgs/Image       mask
sensor_msgs/PointCloud2 cloud
string                  object_class
---
geometry_msgs/Pose      pose
float64                 fitness
float64                 rmse
bool                    success
sensor_msgs/PointCloud2 cutout_cloud
sensor_msgs/PointCloud2 template_cloud
```

### 6.4 Cutout — `concrete_block_perception_interfaces/srv/ExtractMaskCutout`

```
sensor_msgs/Image       mask
sensor_msgs/PointCloud2 cloud
---
bool                    success
string                  reason
sensor_msgs/PointCloud2 cutout_cloud
```

### 6.5 Tracking — `concrete_block_perception_interfaces/srv/TrackDetections`

```
vision_msgs/Detection2DArray detections
sensor_msgs/Image mask
---
concrete_block_perception_interfaces/TrackedDetectionArray tracked
```

### 6.6 `concrete_block_perception_interfaces/msg/TrackedDetection[Array]`

```
# TrackedDetection.msg
uint32 detection_id
vision_msgs/Detection2D detection
sensor_msgs/Image mask
uint32 age
uint32 misses
float32 suppression_radius
builtin_interfaces/Time stamp

# TrackedDetectionArray.msg
builtin_interfaces/Time stamp
TrackedDetection[] detections
```

### 6.7 Orchestrator entry — `concrete_block_world_model_interfaces/srv/RunPoseEstimation`

```
string mode            # SCENE_DISCOVERY | REFINE_BLOCK | REFINE_GRASPED
string target_block_id
bool enable_debug
float32 timeout_s
---
bool success
concrete_block_world_model_interfaces/BlockArray blocks
string message
```

### 6.8 World-model block — `concrete_block_world_model_interfaces/msg/Block[Array]`

```
# Block.msg
int32 POSE_UNKNOWN=0   POSE_COARSE=1   POSE_PRECISE=2
int32 TASK_UNKNOWN=0   TASK_FREE=1  TASK_MOVE=2  TASK_PLACED=3  TASK_REMOVED=4
int32 GOAL_NONE=0      GOAL_SET=1

string id
int32 pose_status
int32 task_status
geometry_msgs/Pose pose
float64[36] pose_covariance     # row-major 6x6 (x,y,z,rx,ry,rz)
float32 confidence
builtin_interfaces/Time last_seen
geometry_msgs/Pose goal_pose
int32 goal_status

# BlockArray.msg
std_msgs/Header header
Block[] blocks
```

### 6.9 Planning scene — `concrete_block_world_model_interfaces/msg/PlanningScene[Object]`

```
# PlanningSceneObject.msg
uint8 SHAPE_UNKNOWN=0  SHAPE_BOX=1
uint8 SOURCE_UNKNOWN=0 SOURCE_BLOCK=1 SOURCE_STATIC_OBSTACLE=2
string id
string frame_id
uint8 shape_type
uint8 source_type
geometry_msgs/Pose pose
geometry_msgs/Vector3 dimensions
int32 pose_status
int32 task_status
float32 confidence

# PlanningScene.msg
std_msgs/Header header
PlanningSceneObject[] objects
```

### 6.10 Other world-model services (request → response)

```
GetCoarseBlocks:   bool force_refresh, float32 timeout_s, builtin_interfaces/Time query_stamp
                   --- bool success, BlockArray blocks, string message

GetPlanningScene:  bool force_refresh, float32 timeout_s, builtin_interfaces/Time query_stamp
                   --- bool success, PlanningScene scene, string message

SetBlockTaskStatus: string block_id, int32 task_status,
                    bool has_grasp_offset=false, geometry_msgs/Pose grasp_offset
                    --- bool success, string message

UpsertBlock:       string block_id, geometry_msgs/Pose pose, string frame_id,
                   int32 pose_status=1, int32 task_status=1, float32 confidence=1.0
                   --- bool success, string message

SetBlockGoal:      string block_id, geometry_msgs/Pose goal_pose, string frame_id
                   --- bool success, string message

ClearBlockGoals:   string[] block_ids  --- bool success, uint32 cleared_count, string message
ClearWorldModel:   (empty)             --- bool success, uint32 cleared_count, string message
```

---

## 7. Data flow per request

**`SCENE_DISCOVERY`**
1. On each synced `(image, points)` pair the orchestrator caches the latest frame.
2. Request arrives → call `/yolos_segmentor_service/segment` → `Detection2DArray` + mask.
3. For each detection: call `/extract_mask_cutout` (mask ∩ cloud → cutout), then send a
   `register_block` action goal (ICP against the CAD template `ConcreteBlock.ply`).
4. Associate each result to an existing world-model block (distance/age gates) or create a
   new one; on registration failure optionally emit a `POSE_COARSE` block from the centroid.
5. Update persistent world model, publish `/cbp/block_world_model` + markers, return `BlockArray`.

**`REFINE_BLOCK`** — same, restricted to the target block; optional pose-seeded ROI.

**`REFINE_GRASPED`** — resolve target (arg or newest `TASK_MOVE`), build FK+ROI around the
TCP→block transform, segment inside ROI, register, then fuse **registration translation +
FK orientation** before writing back.

---

## 8. Building a new detector

Key integration facts for a drop-in replacement or alternative detector:

- **The detector contract is the segmentation service**
  `/yolos_segmentor_service/segment` of type `ros2_yolos_cpp/srv/SegmentImage`
  (request: `image`; response: `detections` `vision_msgs/Detection2DArray` + `mask`
  `sensor_msgs/Image`). Anything that provides this service on that name/type works with the
  orchestrator unchanged. The orchestrator uses **both** `detections` (per-instance bbox/class)
  and the combined `mask`.
- **SAM2 already demonstrates an alternative segmentor** (`sam2_segmentation` package,
  `SegmentAtPoint.srv` = image + (x,y) → mask). Its own srv differs from `SegmentImage`; a
  full swap-in detector should match `SegmentImage` or add an adapter.
- **Downstream expects instance masks**, not just boxes — registration works on
  `mask ∩ point_cloud`. A box-only detector needs a mask stage.
- **Object class string** flows detection → `register_block.object_class`; the CAD template
  is chosen from it. Keep class names consistent with `config/block.names`.
- **Frames/sync**: inputs are `image` + `points`, time-synchronized. A new detector consuming
  the cloud must respect the `max_sync_delta_s` gate and the camera→lidar calibration
  (`calib_blackfly_to_seyond.yaml`).
- **You usually do NOT need to touch the orchestrator, world model, or registration** — swap
  the segmentation service and keep the `SegmentImage` contract.

### Relevant config files
- `concrete_block_perception/config/block_detection_tracking.yaml`
- `concrete_block_perception/config/block_registration.yaml`
- `concrete_block_world_model/config/world_model.yaml`
- `concrete_block_perception/config/{ConcreteBlock.ply, block.names, calib_*.yaml}`

### Key source files
- Orchestrator: `concrete_block_world_model/src/nodes/perception_orchestrator_*.cpp`
- Mode parsing: `concrete_block_world_model/src/utils/world_model_utils.cpp`
- Segmentor service: `ros2_yolos_cpp/src/nodes/` (`yolos_segmentor_service`)
- Registration: `concrete_block_perception/src/nodes/block_registration_node.cpp`
- Launch: `concrete_block_perception/launch/perception.launch.py`
```
