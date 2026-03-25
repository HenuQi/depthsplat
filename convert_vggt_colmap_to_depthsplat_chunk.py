import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pycolmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景目录，需包含 images/ 和 sparse/")
    parser.add_argument("--output_root", type=str, required=True,
                        help="输出数据根目录，例如 datasets/my_vggt")
    parser.add_argument("--stage", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--chunk_name", type=str, default="000000.torch")
    parser.add_argument("--scene_key", type=str, default=None,
                        help="可选，默认使用 scene_dir 名称")
    return parser.parse_args()


def read_image_as_uint8_tensor(path: Path):
    # 与 DepthSplat 原脚本一致，存原始字节
    data = path.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    return torch.from_numpy(arr)


def get_fx_fy_cx_cy_from_colmap_camera(cam):
    # 支持常见 PINHOLE / SIMPLE_PINHOLE / SIMPLE_RADIAL
    params = np.array(cam.params, dtype=np.float64)
    model = str(cam.model)

    if model == "PINHOLE":
        # fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
    elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
        # f, cx, cy, ...
        f, cx, cy = params[:3]
        fx, fy = f, f
    else:
        raise ValueError(f"暂未支持的相机模型: {model}")

    return float(fx), float(fy), float(cx), float(cy)


def image_w2c_3x4(image):
    # pycolmap API 版本略有差异，这里做兼容
    # 常见路径1: image.cam_from_world.matrix()
    # 常见路径2: image.projection_matrix() 再取前3x4（不推荐）
    if hasattr(image, "cam_from_world"):
        mat4 = image.cam_from_world.matrix()
        return np.array(mat4, dtype=np.float64)[:3, :4]
    else:
        # 兜底：尝试通过 qvec/tvec 构造
        if hasattr(image, "qvec") and hasattr(image, "tvec"):
            # qvec 为 [qw, qx, qy, qz]
            qw, qx, qy, qz = image.qvec
            R = np.array([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
            ], dtype=np.float64)
            t = np.array(image.tvec, dtype=np.float64).reshape(3, 1)
            return np.concatenate([R, t], axis=1)
        raise RuntimeError("无法从 pycolmap Image 对象提取 w2c，请检查 pycolmap 版本。")


def build_example(scene_dir: Path, scene_key: str):
    image_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse"

    if not image_dir.exists():
        raise FileNotFoundError(f"缺少目录: {image_dir}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"缺少目录: {sparse_dir}")

    # 读取 COLMAP 重建
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))

    # 按文件名排序，保证 timestamps 与 images 对齐
    items = []
    for image_id, img in reconstruction.images.items():
        img_name = img.name
        img_path = image_dir / img_name
        if not img_path.exists():
            continue

        cam = reconstruction.cameras[img.camera_id]
        width, height = int(cam.width), int(cam.height)
        fx, fy, cx, cy = get_fx_fy_cx_cy_from_colmap_camera(cam)

        # 归一化内参（DepthSplat 约定）
        nfx = fx / width
        nfy = fy / height
        ncx = cx / width
        ncy = cy / height

        w2c = image_w2c_3x4(img).reshape(-1)  # 12

        camera_vec = np.concatenate([
            np.array([nfx, nfy, ncx, ncy, 0.0, 0.0], dtype=np.float32),
            w2c.astype(np.float32),
        ], axis=0)

        # timestamp 用排序下标，保持和 images/timestamps 顺序一致
        items.append((img_name, camera_vec, img_path))

    if len(items) == 0:
        raise RuntimeError("未找到可用图像与重建项。")

    items.sort(key=lambda x: x[0])  # 按文件名排序

    timestamps = torch.arange(len(items), dtype=torch.int64)
    cameras = torch.tensor(np.stack([x[1] for x in items], axis=0), dtype=torch.float32)
    images = [read_image_as_uint8_tensor(x[2]) for x in items]

    example = {
        "url": str(scene_dir),
        "timestamps": timestamps,
        "cameras": cameras,
        "key": scene_key,
        "images": images,
    }
    return example


def main():
    args = parse_args()

    scene_dir = Path(args.scene_dir)
    out_stage = Path(args.output_root) / args.stage
    out_stage.mkdir(parents=True, exist_ok=True)

    scene_key = args.scene_key if args.scene_key else scene_dir.name
    example = build_example(scene_dir, scene_key)

    chunk_path = out_stage / args.chunk_name
    torch.save([example], chunk_path)
    print(f"saved chunk: {chunk_path}")

    index_path = out_stage / "index.json"
    index = {}
    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)

    index[scene_key] = args.chunk_name
    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"updated index: {index_path}")
    print(f"scene_key: {scene_key}")


if __name__ == "__main__":
    main()