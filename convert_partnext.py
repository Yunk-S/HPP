"""
PartNeXt 数据集转换脚本

将 PartNeXt 的原始数据格式转换为 HPP-SAM 可用的 HuggingFace Dataset 格式。

输入格式（本地或远程路径）:
- GLB 文件: {root}/PartNeXt_mesh/glbs/{category_id}/{sample_id}.glb
- JSON 标注: {root}/PartNeXt_raw_json/{category_id}/{sample_id}/{sample_id}.json

或者使用独立路径指定:
- GLB 文件: {glb_root}/{category_id}/{sample_id}.glb
- JSON 标注: {json_root}/{category_id}/{sample_id}/{sample_id}.json

输出格式 (HPP-SAM 兼容):
- coords: [N, 3] float32, 点云坐标
- features: [N, 3] float32, RGB 颜色
- gt_masks: [M, N] bool, 多标签 mask
- depths: [M] 节点深度
- parent_ids: [M] 父节点 ID
- is_leafs: [M] 是否叶子节点
- mask_areas: [M] mask 面积

用法:
    # 使用统一 root
    python convert_partnext.py --root /path/to/PartNeXt --num_points 10000 --num_proc 8

    # 使用独立路径（超算平台）
    python convert_partnext.py --glb_root /path/to/glbs --json_root /path/to/raw_json --num_points 10000

超算平台示例:
    python convert_partnext.py \
        --glb_root /gpfs/work/aac/yunkunshi23/PartNeXt/dataTransform/PartNeXt_mesh/glbs \
        --json_root /gpfs/work/aac/yunkunshi23/PartNeXt/dataTransform/PartNeXt_raw_json \
        --output ./data/partnext \
        --num_points 10000 \
        --num_proc 32
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


def _flatten_int_array(x: Any) -> np.ndarray:
    """
    将 JSON 中的 face id 列表（可能嵌套、浮点、标量）压平为 int64 一维数组。
    """
    if x is None:
        return np.array([], dtype=np.int64)
    if isinstance(x, (int, np.integer)):
        return np.array([int(x)], dtype=np.int64)
    if isinstance(x, float):
        return np.array([int(round(x))], dtype=np.int64)
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    return np.round(arr).astype(np.int64)


def _mesh_face_count(mesh_face_list: dict, mesh_index: int) -> int:
    """mesh_face_num 的 key 可能是 str 或 int。"""
    if mesh_index in mesh_face_list:
        return int(mesh_face_list[mesh_index])
    k = str(mesh_index)
    if k in mesh_face_list:
        return int(mesh_face_list[k])
    raise KeyError(f"mesh_face_num missing index {mesh_index!r}")


def _mesh_to_geometry_only(mesh) -> Any:
    """
    丢弃材质/UV 等 visual，避免 trimesh 在 update_faces/concatenate 时触发 'uv' 相关错误。
    """
    import trimesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False, visual=None)


def _try_load_glb(path: str) -> Any:
    import trimesh

    try:
        return trimesh.load(path, process=False, skip_materials=True)
    except TypeError:
        return trimesh.load(path, process=False)


def _attach_vertex_colors_if_matching(mesh: Any, source: Any) -> None:
    """若 source 与 mesh 顶点数一致，则挂上 ColorVisuals（仅顶点色，无 UV/纹理）。"""
    import trimesh

    vis = getattr(source, "visual", None)
    if vis is None:
        return
    vc = getattr(vis, "vertex_colors", None)
    if vc is None:
        return
    vc = np.asarray(vc)
    if vc.ndim != 2 or vc.shape[0] != len(mesh.vertices):
        return
    try:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vc)
    except Exception:
        pass


# ==============================================================================
# PartNeXt JSON 标注解析
# ==============================================================================

class PartNeXtAnnotationParser:
    """解析 PartNeXt 的 JSON 标注格式"""

    @staticmethod
    def parse_masks(
        annotation_json: dict,
        face_index: np.ndarray,
        num_samples: int
    ) -> Tuple[List[np.ndarray], List[int], List[int], List[int], List[bool]]:
        """
        从 JSON 标注中解析出每个节点的 mask 及粒度信息。

        Args:
            annotation_json: JSON 标注内容
            face_index: 点对应的面索引 [N]
            num_samples: 采样点数

        Returns:
            masks: [M, N] bool 列表
            node_ids: [M] 节点 ID 列表
            depths: [M] 节点深度列表
            parent_ids: [M] 父节点 ID 列表
            is_leafs: [M] 是否叶子节点列表
        """
        masks = []
        node_ids = []
        depths = []
        parent_ids = []
        is_leafs = []

        hierarchy = annotation_json["hierarchyList"][0]
        mesh_face_list = annotation_json["mesh_face_num"]
        masks_dict = annotation_json["masks"]

        # 收集所有节点信息（带深度和父节点）
        all_nodes = []

        def collect_nodes(node, depth=0, parent_id=-1):
            node_id = node["nodeId"]

            # 跳过根节点 (nodeId = 0 or refNodeId = -1)
            if node_id in (0, -1) or node.get("refNodeId") in (0, -1):
                pass
            else:
                # 提取 mask_id_list
                mask_id_list = node.get("maskIdList", node.get("mask_id_list", []))

                all_nodes.append({
                    "node_id": node_id,
                    "depth": depth,
                    "parent_id": parent_id,
                    "is_leaf": "children" not in node or len(node["children"]) == 0,
                    "mask_id_list": mask_id_list,
                })

            if "children" in node:
                for child in node["children"]:
                    collect_nodes(child, depth + 1, node_id)

        collect_nodes(hierarchy)

        for node_info in all_nodes:
            node_id = node_info["node_id"]
            mask_id_list = node_info["mask_id_list"]

            if not mask_id_list:
                continue

            mask_face_id_list = []

            for mask_idx in mask_id_list:
                mask_str = str(mask_idx)
                if mask_str in masks_dict:
                    for mesh_idx_str, face_ids in masks_dict[mask_str].items():
                        mesh_idx = int(mesh_idx_str)
                        offset = sum(
                            _mesh_face_count(mesh_face_list, i)
                            for i in range(mesh_idx)
                        )
                        ids = _flatten_int_array(face_ids) + int(offset)
                        mask_face_id_list.extend(ids.tolist())

            if not mask_face_id_list:
                continue

            mask_face_id_list = np.asarray(mask_face_id_list, dtype=np.int64)
            face_index_i = np.asarray(face_index, dtype=np.int64).ravel()
            mask = np.isin(face_index_i, mask_face_id_list)

            nonzero_count = np.count_nonzero(mask)

            # 跳过全 0 或全 1 的 mask
            if nonzero_count == 0 or nonzero_count == num_samples:
                continue

            masks.append(mask)
            node_ids.append(node_id)
            depths.append(node_info["depth"])
            parent_ids.append(node_info["parent_id"])
            is_leafs.append(node_info["is_leaf"])

        return masks, node_ids, depths, parent_ids, is_leafs

    @staticmethod
    def get_all_node_mask_id_list(hierarchy: dict) -> Dict[int, List[int]]:
        """
        递归收集所有节点的 mask_id 列表（简化版，用于兼容性）。

        Args:
            hierarchy: JSON 中的 hierarchyList[0]

        Returns:
            {nodeId: [mask_id_list]}
        """
        all_node_mask_id_list = {}

        def collect_node_mask_id(node):
            node_id = node["nodeId"]
            # 跳过根节点
            if node_id in (0, -1) or node.get("refNodeId") in (0, -1):
                pass
            else:
                mask_id_list = node.get("maskIdList", node.get("mask_id_list", []))
                all_node_mask_id_list[node_id] = mask_id_list

            if "children" in node:
                for child in node["children"]:
                    collect_node_mask_id(child)

        collect_node_mask_id(hierarchy)
        return all_node_mask_id_list


# ==============================================================================
# GLB 网格采样
# ==============================================================================

def sample_surface_uniform(
    mesh,
    num_samples: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 mesh 表面均匀采样点。

    Args:
        mesh: trimesh.Trimesh 对象
        num_samples: 采样点数
        seed: 随机种子

    Returns:
        points: [N, 3] 采样点坐标
        face_idx: [N] 每个点对应的面索引
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 按面积加权采样
    face_areas = mesh.area_faces
    face_probs = face_areas / face_areas.sum()

    # 采样面索引
    sampled_face_indices = np.random.choice(
        len(mesh.faces),
        size=num_samples,
        replace=True,
        p=face_probs
    )

    # 在每个面上随机采样点
    u = np.random.rand(num_samples, 1)
    v = np.random.rand(num_samples, 1)
    w = 1 - u - v
    valid_mask = (u + v <= 1)

    # 重新采样无效点（布尔掩码对 (N,1) 索引会得到一维视图，RHS 必须一维）
    while not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        ru = np.random.rand(n_invalid)
        rv = np.random.rand(n_invalid)
        u.ravel()[~valid_mask.ravel()] = ru
        v.ravel()[~valid_mask.ravel()] = rv
        w = 1.0 - u - v
        valid_mask = (u + v <= 1)

    # 顶点坐标
    vertices = mesh.vertices[mesh.faces[sampled_face_indices]]

    # 计算采样点坐标
    points = (
        u * vertices[:, 0] +
        v * vertices[:, 1] +
        w * vertices[:, 2]
    ).squeeze()

    return points, sampled_face_indices


def load_and_sample_glb(
    glb_path: str,
    num_samples: int,
    seed: Optional[int] = None,
    with_colors: bool = True
) -> Optional[Dict]:
    """
    加载 GLB 文件并进行表面采样。

    Args:
        glb_path: GLB 文件路径
        num_samples: 采样点数
        seed: 随机种子
        with_colors: 是否提取颜色

    Returns:
        包含 points, colors, face_index 的字典，失败返回 None
    """
    try:
        import trimesh

        loaded = _try_load_glb(glb_path)

        if isinstance(loaded, trimesh.Scene):
            mesh_list_src = [
                g
                for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
            ]
            if not mesh_list_src:
                return None
            mesh_list = [_mesh_to_geometry_only(g) for g in mesh_list_src]
            if len(mesh_list) == 1:
                mesh = mesh_list[0]
                _attach_vertex_colors_if_matching(mesh, mesh_list_src[0])
            else:
                mesh = trimesh.util.concatenate(mesh_list)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = _mesh_to_geometry_only(loaded)
            _attach_vertex_colors_if_matching(mesh, loaded)
        else:
            return None

        # 过滤无效面（geometry-only 不会触发 UV 更新）
        valid_faces = mesh.area_faces > 0
        if not valid_faces.all():
            mesh.update_faces(np.where(valid_faces)[0])

        # 采样
        points, face_idx = sample_surface_uniform(mesh, num_samples, seed)

        result = {
            "points": points.astype(np.float32),
            "face_index": face_idx.astype(np.int64),
        }

        # 提取颜色 (如果有)
        if with_colors and hasattr(mesh, 'visual') and mesh.visual is not None:
            try:
                colors = _extract_colors(mesh, face_idx)
                result["colors"] = colors.astype(np.float32)
            except Exception:
                # 没有颜色时使用默认值
                result["colors"] = np.ones((num_samples, 3), dtype=np.float32) * 0.5
        else:
            result["colors"] = np.ones((num_samples, 3), dtype=np.float32) * 0.5

        return result

    except Exception as e:
        print(f"[WARN] Failed to process {glb_path}: {e}")
        return None


def _extract_colors(mesh, face_idx: np.ndarray) -> np.ndarray:
    """从 mesh 提取颜色"""
    num_samples = len(face_idx)
    fi = np.asarray(face_idx, dtype=np.int64).ravel()
    faces = np.asarray(mesh.faces, dtype=np.int64)[fi]

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # 顶点颜色
        vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64)
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
        colors = vertex_colors[faces].mean(axis=1)
    elif hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        # 材质颜色
        if hasattr(mesh.visual.material, 'baseColorFactor'):
            color = np.array(mesh.visual.material.baseColorFactor[:3])
        else:
            color = np.array([0.8, 0.8, 0.8])
        colors = np.tile(color, (num_samples, 1))
    else:
        colors = np.ones((num_samples, 3), dtype=np.float32) * 0.8

    return colors


# ==============================================================================
# 数据集构建
# ==============================================================================

def build_partnext_index(
    root: Optional[Path] = None,
    glb_root: Optional[Path] = None,
    json_root: Optional[Path] = None,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    构建 PartNeXt 数据集索引。

    支持两种模式：
    1. 统一 root: --root /path/to/PartNeXt
       (会自动查找 PartNeXt_mesh/glbs 和 PartNeXt_raw_json)
    2. 独立路径: --glb_root /path/to/glbs --json_root /path/to/raw_json

    Args:
        root: PartNeXt 数据根目录（模式1）
        glb_root: GLB 文件根目录（模式2）
        json_root: JSON 标注根目录（模式2）
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (train_items, test_items) - 样本列表
    """
    # 确定路径
    if root is not None:
        glb_base = root / "PartNeXt_mesh" / "glbs"
        json_base = root / "PartNeXt_raw_json"
    elif glb_root is not None and json_root is not None:
        glb_base = Path(glb_root)
        json_base = Path(json_root)
    else:
        raise ValueError("Must provide either --root or both --glb_root and --json_root")

    if not glb_base.exists():
        raise FileNotFoundError(f"GLB folder not found: {glb_base}")
    if not json_base.exists():
        raise FileNotFoundError(f"Annotation folder not found: {json_base}")

    all_items = []

    type_dirs = sorted([p for p in glb_base.iterdir() if p.is_dir()])

    for type_dir in type_dirs:
        type_id = type_dir.name
        json_type_dir = json_base / type_id

        if not json_type_dir.exists():
            print(f"[WARN] Missing annotation folder for {type_id}")
            continue

        glb_files = sorted(type_dir.glob("*.glb"))
        for glb_path in glb_files:
            glb_id = glb_path.stem
            json_path = json_type_dir / glb_id / f"{glb_id}.json"

            if json_path.exists():
                all_items.append({
                    "glb_id": glb_id,
                    "type_id": type_id,
                    "glb_path": str(glb_path),
                    "json_path": str(json_path),
                })

    print(f"Matched samples: {len(all_items)}")

    # 随机划分
    random.seed(seed)
    random.shuffle(all_items)

    n = len(all_items)
    n_train = int(n * train_ratio)

    train_items = all_items[:n_train]
    test_items = all_items[n_train:]

    return train_items, test_items


def process_single_sample(
    item: dict,
    num_points: int,
    seed_offset: int = 0
) -> Optional[dict]:
    """
    处理单个 PartNeXt 样本。

    Args:
        item: 样本信息字典
        num_points: 采样点数
        seed_offset: 随机种子偏移

    Returns:
        处理后的样本数据，失败返回 None
    """
    # 采样点云
    sampled = load_and_sample_glb(
        item["glb_path"],
        num_samples=num_points,
        seed=hash(item["glb_id"]) % (2**31) + seed_offset
    )

    if sampled is None:
        return None

    # 解析标注
    with open(item["json_path"], 'r') as f:
        annotation = json.load(f)

    parser = PartNeXtAnnotationParser()
    masks, node_ids, depths, parent_ids, is_leafs = parser.parse_masks(
        annotation,
        sampled["face_index"],
        num_points
    )

    if len(masks) == 0:
        return None

    # 计算 mask areas
    mask_areas = np.array([np.count_nonzero(m) for m in masks], dtype=np.int64)

    return {
        "coords": sampled["points"],
        "features": sampled["colors"],
        "gt_masks": np.array(masks),
        "sample_id": item["glb_id"],
        "type_id": item["type_id"],
        "node_ids": np.array(node_ids),
        "depths": np.array(depths, dtype=np.int64),
        "parent_ids": np.array(parent_ids, dtype=np.int64),
        "is_leafs": np.array(is_leafs),
        "mask_areas": mask_areas,
    }


def create_hf_dataset(
    items: List[dict],
    num_points: int,
    num_proc: int = 1,
    desc: str = "Processing"
) -> Dataset:
    """
    创建 HuggingFace Dataset。

    Args:
        items: 样本列表
        num_points: 采样点数
        num_proc: 并行进程数
        desc: 进度条描述

    Returns:
        HuggingFace Dataset
    """
    from datasets import Features, Array3D, Sequence, Value

    def process_wrapper(item):
        return process_single_sample(item, num_points)

    # 序列化 items 为字符串以便并行处理
    items_json = json.dumps(items)

    def process_from_json(json_str):
        items = json.loads(json_str)
        results = []
        for item in tqdm(items, desc=desc):
            result = process_single_sample(item, num_points)
            if result is not None:
                results.append(result)
        return results

    # 使用 map 风格处理
    processed = []

    for item in tqdm(items, desc=desc):
        result = process_single_sample(item, num_points)
        if result is not None:
            processed.append(result)

    # 定义特征
    features = Features({
        "coords": Array3D(dtype="float32", shape=(num_points, 3)),
        "features": Array3D(dtype="float32", shape=(num_points, 3)),
        "gt_masks": Sequence(
            feature=Array3D(dtype="bool", shape=(num_points,)),
            length=-1,
        ),
        "sample_id": Value(dtype="string"),
        "type_id": Value(dtype="string"),
        "node_ids": Sequence(feature=Value(dtype="int64"), length=-1),
        "depths": Sequence(feature=Value(dtype="int64"), length=-1),
        "parent_ids": Sequence(feature=Value(dtype="int64"), length=-1),
        "is_leafs": Sequence(feature=Value(dtype="bool"), length=-1),
        "mask_areas": Sequence(feature=Value(dtype="int64"), length=-1),
    })

    return Dataset.from_list(processed, features=features)


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert PartNeXt dataset to HPP-SAM format"
    )

    # 输入路径参数（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--root",
        type=str,
        help="Root directory of PartNeXt dataset (with PartNeXt_mesh/glbs and PartNeXt_raw_json subdirs)"
    )
    input_group.add_argument(
        "--glb_root",
        type=str,
        help="Path to GLB files directory (with category subdirs)"
    )

    parser.add_argument(
        "--json_root",
        type=str,
        default=None,
        help="Path to JSON annotations directory (required with --glb_root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./data/partnext)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=10000,
        help="Number of points to sample per object (default: 10000)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Training set ratio (default: 0.9)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["small", "full"],
        help="Mode: small (50 train + 10 test) or full (default: full)"
    )

    args = parser.parse_args()

    # 确定输入路径
    if args.root:
        input_root = Path(args.root)
        glb_root = None
        json_root = None
    else:
        input_root = None
        glb_root = Path(args.glb_root)
        json_root = Path(args.json_root) if args.json_root else None

    # 确定输出路径
    output_dir = Path(args.output) if args.output else Path("./data/partnext")

    print("=" * 60)
    print("PartNeXt to HPP-SAM Dataset Converter")
    print("=" * 60)
    if input_root:
        print(f"Root:       {input_root}")
    else:
        print(f"GLB root:   {glb_root}")
        print(f"JSON root:  {json_root}")
    print(f"Output:     {output_dir}")
    print(f"Num points: {args.num_points}")
    print(f"Num proc:   {args.num_proc}")
    print(f"Train ratio:{args.train_ratio}")
    print(f"Mode:       {args.mode}")
    print("=" * 60)

    # 构建索引
    print("\n[1/4] Building dataset index...")
    train_items, test_items = build_partnext_index(
        root=input_root,
        glb_root=glb_root,
        json_root=json_root,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    # 限制 small 模式
    if args.mode == "small":
        train_items = train_items[:50]
        test_items = test_items[:10]
        print(f"Small mode: using {len(train_items)} train, {len(test_items)} test")

    print(f"Train: {len(train_items)} | Test: {len(test_items)}")

    # 处理数据集
    print(f"\n[2/4] Processing training set ({len(train_items)} samples)...")
    train_dataset = create_hf_dataset(
        train_items,
        num_points=args.num_points,
        num_proc=args.num_proc,
        desc="Train"
    )

    print(f"\n[3/4] Processing test set ({len(test_items)} samples)...")
    test_dataset = create_hf_dataset(
        test_items,
        num_points=args.num_points,
        num_proc=args.num_proc,
        desc="Test"
    )

    # 保存
    print(f"\n[4/4] Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    # 重命名列为 HPP-SAM 格式
    dataset_dict = dataset_dict.rename_columns({
        "xyz": "coords",
        "rgb": "features",
        "mask": "gt_masks",
    })

    # 保存
    dataset_path = output_dir / "dataset"
    dataset_dict.save_to_disk(str(dataset_path))

    # 保存元信息
    meta = {
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "num_points": args.num_points,
        "categories": list(set(item["type_id"] for item in train_items + test_items)),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Dataset saved to: {dataset_path}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Points per sample: {args.num_points}")
    print(f"Categories: {len(meta['categories'])}")

    # 验证
    print("\n[Verify] Sample data:")
    sample = train_dataset[0]
    print(f"  coords shape: {np.array(sample['coords']).shape}")
    print(f"  features shape: {np.array(sample['features']).shape}")
    print(f"  gt_masks: {len(sample['gt_masks'])} masks")
    print(f"  sample_id: {sample['sample_id']}")


if __name__ == "__main__":
    main()
