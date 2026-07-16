#!/usr/bin/env python3
"""Export RGB images, point clouds, and TF messages from a ROS 2 bag.

Run inside the ROS workspace after sourcing install/setup.bash, for example:

  python3 tools/export_reference_bag.py \
    /path/to/bag_dir \
    --output reference_bag_export \
    --point-topic /seyond/points \
    --image-topic /blackfly_rotated/image_rect \
    --camera-info-topic /blackfly_rotated/camera_info
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def msg_stamp_ns(msg: Any, fallback_ns: int) -> int:
    header = getattr(msg, "header", None)
    if header is not None:
        return stamp_to_ns(header.stamp)
    return fallback_ns


def ensure_dirs(root: Path) -> None:
    for name in ["images", "pointclouds", "tf"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def open_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    storage_id = "sqlite3"
    if any(bag_path.glob("*.mcap")):
        storage_id = "mcap"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def write_camera_info(path: Path, msg: Any) -> None:
    data = {
        "header": {
            "frame_id": msg.header.frame_id,
            "stamp_ns": stamp_to_ns(msg.header.stamp),
        },
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": msg.distortion_model,
        "D": list(msg.d),
        "K": list(msg.k),
        "R": list(msg.r),
        "P": list(msg.p),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_tf_jsonl(path: Path, msg: Any, bag_time_ns: int) -> None:
    with path.open("a", encoding="utf-8") as f:
        for tr in msg.transforms:
            t = tr.transform.translation
            q = tr.transform.rotation
            rec = {
                "bag_time_ns": bag_time_ns,
                "stamp_ns": stamp_to_ns(tr.header.stamp),
                "parent": tr.header.frame_id,
                "child": tr.child_frame_id,
                "translation": [t.x, t.y, t.z],
                "rotation_xyzw": [q.x, q.y, q.z, q.w],
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def image_to_file(root: Path, msg: Any, index: int, fallback_ns: int) -> dict[str, Any]:
    stamp_ns = msg_stamp_ns(msg, fallback_ns)
    frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
    msg_type = type(msg).__module__ + "." + type(msg).__name__

    if msg_type.endswith("CompressedImage"):
        fmt = (msg.format or "jpg").lower()
        ext = "png" if "png" in fmt else "jpg"
        out = root / "images" / f"{index:06d}_{stamp_ns}.{ext}"
        out.write_bytes(bytes(msg.data))
        return {
            "index": index,
            "stamp_ns": stamp_ns,
            "frame_id": frame_id,
            "path": str(out.relative_to(root)),
            "encoding": msg.format,
            "compressed": True,
        }

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Raw Image export needs cv2 and numpy installed") from exc

    enc = msg.encoding.lower()
    channels = 1
    dtype = np.uint8
    if enc in ("rgb8", "bgr8"):
        channels = 3
    elif enc in ("rgba8", "bgra8"):
        channels = 4
    elif enc in ("mono16", "16uc1"):
        dtype = np.uint16
    else:
        raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")

    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        arr = arr.reshape((msg.height, msg.step // arr.itemsize))[:, : msg.width]
    else:
        arr = arr.reshape((msg.height, msg.step // arr.itemsize // channels, channels))[:, : msg.width, :]

    if enc == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif enc == "rgba8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

    out = root / "images" / f"{index:06d}_{stamp_ns}.png"
    cv2.imwrite(str(out), arr)
    return {
        "index": index,
        "stamp_ns": stamp_ns,
        "frame_id": frame_id,
        "path": str(out.relative_to(root)),
        "encoding": msg.encoding,
        "compressed": False,
        "width": int(msg.width),
        "height": int(msg.height),
    }


def pointcloud_fields(msg: Any) -> dict[str, Any]:
    return {field.name: field for field in msg.fields}


def field_unpack(fmt: str, data: bytes, offset: int) -> Any:
    return struct.unpack_from(fmt, data, offset)[0]


def datatype_fmt(datatype: int) -> str | None:
    # sensor_msgs/PointField constants
    return {
        1: "b",
        2: "B",
        3: "h",
        4: "H",
        5: "i",
        6: "I",
        7: "f",
        8: "d",
    }.get(datatype)


def write_pcd(root: Path, msg: Any, index: int, fallback_ns: int) -> dict[str, Any]:
    stamp_ns = msg_stamp_ns(msg, fallback_ns)
    frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
    fields = pointcloud_fields(msg)
    required = ["x", "y", "z"]
    for name in required:
        if name not in fields:
            raise RuntimeError(f"PointCloud2 is missing required field {name}")

    optional_names = [n for n in ["intensity", "rgb", "rgba"] if n in fields]
    names = required + optional_names
    rows: list[list[float]] = []
    endian = ">" if msg.is_bigendian else "<"
    data = bytes(msg.data)

    for row in range(msg.height):
        base = row * msg.row_step
        for col in range(msg.width):
            pbase = base + col * msg.point_step
            values = []
            skip = False
            for name in names:
                field = fields[name]
                fmt = datatype_fmt(field.datatype)
                if fmt is None:
                    continue
                value = field_unpack(endian + fmt, data, pbase + field.offset)
                if name in required and not math.isfinite(float(value)):
                    skip = True
                values.append(float(value))
            if not skip:
                rows.append(values)

    out = root / "pointclouds" / f"{index:06d}_{stamp_ns}.pcd"
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS " + " ".join(names),
        "SIZE " + " ".join(["4"] * len(names)),
        "TYPE " + " ".join(["F"] * len(names)),
        "COUNT " + " ".join(["1"] * len(names)),
        f"WIDTH {len(rows)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(rows)}",
        "DATA ascii",
    ]
    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for values in rows:
            f.write(" ".join(f"{v:.9g}" for v in values) + "\n")

    return {
        "index": index,
        "stamp_ns": stamp_ns,
        "frame_id": frame_id,
        "path": str(out.relative_to(root)),
        "fields": names,
        "points": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference_bag_export"))
    parser.add_argument("--point-topic", default="/seyond/points")
    parser.add_argument("--image-topic", default="/blackfly_rotated/image_rect")
    parser.add_argument("--camera-info-topic", default="/blackfly_rotated/camera_info")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--max-frames", type=int, default=0, help="0 exports all clouds/images")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth cloud/image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output
    ensure_dirs(out)

    reader = open_reader(args.bag)
    types = topic_type_map(reader)
    msg_types = {topic: get_message(type_name) for topic, type_name in types.items()}

    wanted = {
        args.point_topic,
        args.image_topic,
        args.camera_info_topic,
        args.tf_topic,
        args.tf_static_topic,
    }
    missing = sorted(topic for topic in wanted if topic not in types)
    if missing:
        print("Warning: topics not present in bag:", ", ".join(missing))

    image_records = []
    cloud_records = []
    image_seen = cloud_seen = 0
    image_written = cloud_written = 0
    camera_info_written = False

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        if topic not in wanted:
            continue
        msg = deserialize_message(data, msg_types[topic])

        if topic == args.tf_topic:
            write_tf_jsonl(out / "tf" / "tf.jsonl", msg, bag_time_ns)
        elif topic == args.tf_static_topic:
            write_tf_jsonl(out / "tf" / "tf_static.jsonl", msg, bag_time_ns)
        elif topic == args.camera_info_topic and not camera_info_written:
            write_camera_info(out / "camera_info.yaml", msg)
            camera_info_written = True
        elif topic == args.image_topic:
            image_seen += 1
            if (image_seen - 1) % args.stride == 0 and (args.max_frames <= 0 or image_written < args.max_frames):
                image_records.append(image_to_file(out, msg, image_written, bag_time_ns))
                image_written += 1
        elif topic == args.point_topic:
            cloud_seen += 1
            if (cloud_seen - 1) % args.stride == 0 and (args.max_frames <= 0 or cloud_written < args.max_frames):
                cloud_records.append(write_pcd(out, msg, cloud_written, bag_time_ns))
                cloud_written += 1

    manifest = {
        "bag": str(args.bag),
        "topics": {
            "point_topic": args.point_topic,
            "image_topic": args.image_topic,
            "camera_info_topic": args.camera_info_topic,
            "tf_topic": args.tf_topic,
            "tf_static_topic": args.tf_static_topic,
        },
        "counts": {
            "images_seen": image_seen,
            "images_exported": image_written,
            "clouds_seen": cloud_seen,
            "clouds_exported": cloud_written,
        },
        "images": image_records,
        "pointclouds": cloud_records,
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(manifest["counts"], sort_keys=False))


if __name__ == "__main__":
    main()
