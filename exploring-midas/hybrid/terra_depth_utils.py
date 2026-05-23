# terra_depth_utils.py

import os
import glob
from typing import Dict, Tuple, Optional

import numpy as np
import re

try:
    import open3d as o3d
except ImportError as _e:
    raise ImportError(
        "terra_depth_utils.py requires open3d. Please install it with "
        "`pip install open3d` before using Terra-based depth."
    ) from _e

# Cache of RaycastingScene per (terra_root, short_scene_name)
_TERRA_SCENES: Dict[Tuple[str, str], "o3d.t.geometry.RaycastingScene"] = {}


def _get_short_scene_name(scene_name: str) -> str:
    """
    Convert a dataset scene name like 'interval1_AMtown01-007' to a stable
    Terra key like 'AMtown' that matches the Terra directory name:

        terra_3dmap_pointcloud_mesh/
            AMtown/
            AMvalley/
            ...

    """
    # Drop the interval prefix: 'interval1_AMtown01-007' -> 'AMtown01-007'
    parts = scene_name.split("_", 1)
    short = parts[1] if len(parts) > 1 else scene_name

    # Drop any trailing frame index part after '-': 'AMtown01-007' -> 'AMtown01'
    short = short.split("-")[0]

    # Drop trailing digits so 'AMtown01' -> 'AMtown', matching your Terra dir
    short = re.sub(r"\d+$", "", short)

    return short


def _load_terra_scene(short_scene: str, terra_root: str) -> "o3d.t.geometry.RaycastingScene":
    """
    Load a Terra mesh for the given short scene name (e.g. 'AMtown') and wrap
    it in a RaycastingScene for fast ray queries.

    Expected layout:

        terra_root/
            AMtown/
                Mesh.ply
                cloud_merged.ply
                terraply/
                    BlockAB/...
            AMvalley/
                Mesh.ply
                ...

    """
    key = (terra_root, short_scene)
    if key in _TERRA_SCENES:
        return _TERRA_SCENES[key]

    scene_dir = os.path.join(terra_root, short_scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(
            f"[Terra depth] No Terra directory for scene '{short_scene}': {scene_dir}"
        )

    mesh_path = None

    # Prefer these canonical files if they exist
    for name in ("Mesh.ply", "cloud_merged.ply"):
        cand = os.path.join(scene_dir, name)
        if os.path.isfile(cand):
            mesh_path = cand
            break

    # Fallback: any .ply directly under scene_dir
    if mesh_path is None:
        candidates = glob.glob(os.path.join(scene_dir, "*.ply"))
        if not candidates:
            raise FileNotFoundError(
                f"[Terra depth] No .ply meshes found in {scene_dir}"
            )
        mesh_path = sorted(candidates)[0]

    print(f"[Terra depth] Loading mesh for '{short_scene}' from {mesh_path}")

    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_legacy.is_empty():
        raise RuntimeError(f"[Terra depth] Empty mesh loaded from {mesh_path}")
    mesh_legacy.compute_vertex_normals()

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    _TERRA_SCENES[key] = scene
    return scene


def raycast_terra_depth_tile(
    scene_name: str,
    terra_root: str,
    K: np.ndarray,
    R_cam_world: np.ndarray,
    t_cam_world: np.ndarray,
    full_W: int,
    full_H: int,
    x0: int,
    y0: int,
    tile_W: int,
    tile_H: int,
    max_depth: float = 200.0,
) -> Optional[np.ndarray]:
    """
    Ray-cast a depth map for a single tile using the Terra 3D mesh.

    Args:
        scene_name: scene identifier from the dataset (e.g. 'interval1_AMtown01-007').
        terra_root: root directory containing Terra meshes, with one subdir per scene.
        K: (3,3) camera intrinsic matrix.
        R_cam_world: (3,3) rotation from world to camera coordinates.
        t_cam_world: (3,) translation from world to camera coordinates.
        full_W, full_H: original image width/height.
        x0, y0: top-left pixel of the tile in full-image coordinates.
        tile_W, tile_H: tile size in pixels.
        max_depth: clamp / reject hits beyond this range (meters).

    Returns:
        depth_tile: (tile_H, tile_W) float32 array with z-depth in camera coordinates,
                    or None if raycasting completely fails.
    """
    if terra_root is None:
        return None

    short = _get_short_scene_name(scene_name)
    scene = _load_terra_scene(short, terra_root)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Generate grid of pixel centers in the full image, restricted to this tile.
    xs = np.arange(x0, x0 + tile_W, dtype=np.float32)
    ys = np.arange(y0, y0 + tile_H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)  # shape [tile_H, tile_W]

    # Directions in camera coordinates (before normalization).
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(x_cam, dtype=np.float32)

    # Normalize to unit directions.
    norm = np.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)  # [N,3]

    # Camera center in world coordinates: C_w = -R^T t
    R = R_cam_world
    t = t_cam_world.reshape(3,)
    R_wc = R.T
    cam_center_world = -R_wc @ t  # [3]

    # Directions in world coordinates
    dirs_world = (R_wc @ dirs_cam.T).T  # [N,3]

    # Build ray buffer [ox, oy, oz, dx, dy, dz]
    N = dirs_world.shape[0]
    origins = np.repeat(cam_center_world[None, :], N, axis=0)
    rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)

    query_rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(query_rays)
    t_hit = ans["t_hit"].numpy().reshape(tile_H, tile_W)  # distance along ray in meters

    # Use distance along the ray directly (same convention as viz_terra_depth_debug)
    depth = t_hit.astype(np.float32)

    invalid = np.isinf(t_hit) | (t_hit <= 0) | (depth <= 0) | (depth >= max_depth)
    depth[invalid] = 0.0
    return depth
