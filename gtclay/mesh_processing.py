"""Mesh loading and segmentation utilities using layer-based top detection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - fallback when SciPy is unavailable
    ConvexHull = None  # type: ignore[assignment]


@dataclass
class SegmentMeshes:
    """Container for separated mesh parts."""

    original: trimesh.Trimesh
    bottom: Optional[trimesh.Trimesh]
    body: Optional[trimesh.Trimesh]
    top: Optional[trimesh.Trimesh]
    bottom_faces: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    body_faces: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    top_faces: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


class MeshProcessingError(RuntimeError):
    """Raised when mesh processing fails."""


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh ensuring it is triangulated and cleaned up."""

    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise MeshProcessingError(f"Unsupported mesh type for {path}")
    if mesh.is_empty:
        raise MeshProcessingError(f"Mesh {path} has no geometry")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.fix_normals()
    return mesh


def _detect_bottom_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return indices of faces that belong to the base at z~0."""

    vertices = mesh.vertices
    faces = mesh.faces
    face_z = vertices[faces][:, :, 2]
    face_centers = mesh.triangles_center

    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    height = max(z_max - z_min, 1e-6)

    tol = max(height * 0.0002, 1e-4)

    face_min = face_z.min(axis=1)
    face_max = face_z.max(axis=1)
    candidate_mask = (face_min >= z_min - tol) & (face_max <= z_min + tol)
    candidate_idx = np.where(candidate_mask)[0]
    if candidate_idx.size == 0:
        return candidate_idx

    neighbors = _build_face_adjacency_list(mesh)
    visited = np.zeros(len(faces), dtype=bool)
    components: list[np.ndarray] = []

    for face_idx in candidate_idx:
        if visited[face_idx] or not candidate_mask[face_idx]:
            continue
        queue: deque[int] = deque([int(face_idx)])
        visited[face_idx] = True
        comp: list[int] = []
        while queue:
            current = queue.popleft()
            comp.append(current)
            for nb in neighbors[current]:
                if visited[nb] or not candidate_mask[nb]:
                    continue
                visited[nb] = True
                queue.append(nb)
        components.append(np.array(comp, dtype=int))

    if not components:
        return candidate_idx

    def component_score(comp: np.ndarray) -> tuple[float, float]:
        centers = face_centers[comp, 2]
        return (centers.max(), -len(comp))

    best_component = min(components, key=component_score)
    return np.asarray(best_component, dtype=int)


def _build_face_adjacency_list(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Return adjacency list for faces."""

    n_faces = len(mesh.faces)
    neighbors: list[list[int]] = [[] for _ in range(n_faces)]
    if mesh.face_adjacency.shape[0] == 0:
        return neighbors
    for a, b in mesh.face_adjacency:
        neighbors[a].append(b)
        neighbors[b].append(a)
    return neighbors


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid and unit normal of the best-fit plane."""

    centroid = points.mean(axis=0)
    if points.shape[0] < 3:
        return centroid, np.array([0.0, 0.0, 1.0])

    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return centroid, np.array([0.0, 0.0, 1.0])

    normal = vh[-1]
    if normal[2] < 0.0:
        normal *= -1.0
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        return centroid, np.array([0.0, 0.0, 1.0])
    return centroid, normal / norm


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal axes spanning the plane perpendicular to normal."""

    normal = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref.dot(normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, ref)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return True if point lies inside the polygon (ray casting)."""

    if polygon.shape[0] < 3:
        return False
    x, y = point
    inside = False
    for i in range(polygon.shape[0]):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % polygon.shape[0]]
        if (y1 > y) != (y2 > y):
            denom = (y2 - y1) if (y2 - y1) != 0.0 else 1e-12
            x_cross = (x2 - x1) * (y - y1) / denom + x1
            if x < x_cross:
                inside = not inside
    return inside


def _signed_angle(base: np.ndarray, vec: np.ndarray) -> float:
    """Return signed 2D angle from base to vec."""

    cross = base[0] * vec[1] - base[1] * vec[0]
    dot = base.dot(vec)
    return math.atan2(cross, dot)


def _trace_rim_path(
    start: int,
    coords2d: np.ndarray,
    candidate_mask: np.ndarray,
    vertex_neighbors: list[np.ndarray],
    center_2d: np.ndarray,
    max_steps: int,
    direction_sign: float,
) -> tuple[list[int], bool]:
    """Trace a closed rim path minimizing the turning angle."""

    path: list[int] = [int(start)]
    prev = None
    current = int(start)

    start_vec = coords2d[current] - center_2d
    if np.linalg.norm(start_vec) < 1e-8:
        start_vec = np.array([1.0, 0.0])
    tangent = direction_sign * np.array([-start_vec[1], start_vec[0]])
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.array([0.0, 1.0])
    tangent = tangent / np.linalg.norm(tangent)
    prev_dir = tangent

    steps = 0
    while steps < max_steps:
        steps += 1
        best_choice = None
        best_angle = None
        current_pt = coords2d[current]
        for nb in vertex_neighbors[current]:
            if not candidate_mask[nb]:
                continue
            if prev is not None and nb == prev:
                continue
            if nb in path and nb != start:
                continue
            vec = coords2d[nb] - current_pt
            length = np.linalg.norm(vec)
            if length < 1e-8:
                continue
            vec /= length
            angle = _signed_angle(prev_dir, vec)
            if angle <= 1e-6:
                angle += 2.0 * math.pi
            if best_angle is None or angle < best_angle:
                best_angle = angle
                best_choice = int(nb)
        if best_choice is None:
            break
        if best_choice == start and len(path) >= 4:
            return path, True
        if best_choice in path:
            break
        path.append(best_choice)
        prev = current
        current = best_choice
        prev_dir = coords2d[current] - coords2d[prev]
        norm_prev = np.linalg.norm(prev_dir)
        if norm_prev < 1e-8:
            break
        prev_dir /= norm_prev
    return path, False


def _detect_top_layer_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Detectar faces ligadas a borda superior (ultimo layer)."""

    vertices = mesh.vertices
    faces = mesh.faces
    face_vertices = vertices[faces]
    z_values = vertices[:, 2]

    z_max = float(z_values.max())
    z_min = float(z_values.min())
    height = max(z_max - z_min, 1e-6)

    unique_z = np.unique(np.round(z_values, 6))
    if unique_z.size > 1:
        diffs = np.diff(unique_z)
        diffs = diffs[diffs > 0]
        if diffs.size:
            window = diffs[-min(200, diffs.size):]
            layer_height = float(np.percentile(window, 75))
        else:
            layer_height = max(height * 0.01, 0.05)
    else:
        layer_height = max(height * 0.01, 0.05)

    band_vertices = max(layer_height * 3.0, 0.5)
    candidate_vertices = np.where(z_values >= z_max - band_vertices)[0]
    if candidate_vertices.size < 3:
        fallback_index = int(np.argmax(face_vertices[:, :, 2].max(axis=1)))
        return np.array([fallback_index], dtype=int)

    candidate_points = vertices[candidate_vertices]
    plane_origin, plane_normal = _fit_plane(candidate_points)
    axis_u, axis_v = _plane_axes(plane_normal)

    offsets = vertices - plane_origin
    coords2d = np.column_stack((offsets @ axis_u, offsets @ axis_v))

    signed_height = offsets @ plane_normal
    plane_band = max(layer_height * 1.5, 0.4)
    plane_mask = np.abs(signed_height) <= plane_band
    plane_indices = np.where(plane_mask)[0]

    rim_mask = np.zeros(len(vertices), dtype=bool)
    for vidx in plane_indices:
        for nb in mesh.vertex_neighbors[vidx]:
            if signed_height[nb] < -plane_band:
                rim_mask[vidx] = True
                break

    candidate_indices = np.where(rim_mask)[0]
    if candidate_indices.size < 3:
        candidate_indices = plane_indices
    if candidate_indices.size < 3:
        fallback_index = int(np.argmax(face_vertices[:, :, 2].max(axis=1)))
        return np.array([fallback_index], dtype=int)

    candidate_mask = np.zeros(len(vertices), dtype=bool)
    candidate_mask[candidate_indices] = True
    ring_center = coords2d[candidate_indices].mean(axis=0)

    start_vertex = int(candidate_indices[np.argmax(z_values[candidate_indices])])
    max_steps = max(len(candidate_indices) * 3, 30)

    path, closed = _trace_rim_path(
        start_vertex,
        coords2d,
        candidate_mask,
        mesh.vertex_neighbors,
        ring_center,
        max_steps,
        1.0,
    )
    use_traced = closed and len(path) >= 4
    if (not use_traced) and candidate_indices.size >= 3:
        path_alt, closed_alt = _trace_rim_path(
            start_vertex,
            coords2d,
            candidate_mask,
            mesh.vertex_neighbors,
            ring_center,
            max_steps,
            -1.0,
        )
        if closed_alt and len(path_alt) >= 4:
            path, closed = path_alt, closed_alt
            use_traced = True

    if use_traced:
        rim_vertices = np.array(path, dtype=int)
    else:
        candidate_coords = coords2d[candidate_indices]
        if candidate_coords.shape[0] >= 3 and ConvexHull is not None:
            try:
                hull = ConvexHull(candidate_coords)
                rim_vertices = candidate_indices[hull.vertices]
            except Exception:
                rim_vertices = candidate_indices
        else:
            fallback_mask = face_vertices[:, :, 2].max(axis=1) >= z_max - band_vertices
            fallback_idx = np.where(fallback_mask)[0]
            if fallback_idx.size == 0:
                fallback_index = int(np.argmax(face_vertices[:, :, 2].max(axis=1)))
                return np.array([fallback_index], dtype=int)
            return fallback_idx

    rim_vertices = np.unique(rim_vertices)
    if rim_vertices.size < 3:
        fallback_mask = face_vertices[:, :, 2].max(axis=1) >= z_max - band_vertices
        fallback_idx = np.where(fallback_mask)[0]
        if fallback_idx.size == 0:
            fallback_index = int(np.argmax(face_vertices[:, :, 2].max(axis=1)))
            return np.array([fallback_index], dtype=int)
        return fallback_idx

    rim_coords = coords2d[rim_vertices]
    center_rim = rim_coords.mean(axis=0)
    angles = np.arctan2(rim_coords[:, 1] - center_rim[1], rim_coords[:, 0] - center_rim[0])
    order = np.argsort(angles)
    rim_vertices = rim_vertices[order]
    rim_coords = rim_coords[order]

    start_offset = int(np.argmax(z_values[rim_vertices]))
    rim_vertices = np.roll(rim_vertices, -start_offset)
    rim_coords = np.roll(rim_coords, -start_offset, axis=0)

    polygon = np.vstack([rim_coords, rim_coords[0]])

    rim_points = vertices[rim_vertices]
    plane_origin, plane_normal = _fit_plane(rim_points)
    axis_u, axis_v = _plane_axes(plane_normal)

    plane_tol = max(layer_height * 2.0, 0.4)
    angle_threshold = math.cos(math.radians(45.0))
    z_cut = z_max - band_vertices * 1.2

    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    selected_mask = np.zeros(len(faces), dtype=bool)
    for idx in range(len(faces)):
        centroid = face_centers[idx]
        if centroid[2] < z_cut:
            continue
        if plane_mask[faces[idx]].sum() < 2:
            continue
        dist = abs((centroid - plane_origin).dot(plane_normal))
        if dist > plane_tol:
            continue
        if face_normals[idx].dot(plane_normal) < angle_threshold:
            continue
        centroid_2d = np.array(
            [(centroid - plane_origin).dot(axis_u), (centroid - plane_origin).dot(axis_v)]
        )
        if not _point_in_polygon(centroid_2d, polygon):
            continue
        selected_mask[idx] = True

    selected = np.where(selected_mask)[0]
    if selected.size >= 3:
        return selected

    fallback_mask = face_vertices[:, :, 2].max(axis=1) >= z_max - band_vertices
    fallback_idx = np.where(fallback_mask)[0]
    if fallback_idx.size == 0:
        fallback_index = int(np.argmax(face_vertices[:, :, 2].max(axis=1)))
        return np.array([fallback_index], dtype=int)
    return fallback_idx


def segment_mesh(path: Path) -> SegmentMeshes:
    """Load and split a mesh into bottom, body, and top."""

    mesh = load_mesh(path)
    bottom_idx = _detect_bottom_faces(mesh)
    top_idx = _detect_top_layer_faces(mesh)

    n_faces = len(mesh.faces)
    body_mask = np.ones(n_faces, dtype=bool)
    body_mask[bottom_idx] = False
    if top_idx.size:
        body_mask[top_idx] = False

    body_idx = np.where(body_mask)[0]

    bottom_mesh = mesh.submesh([bottom_idx], append=True) if bottom_idx.size else None
    body_mesh = mesh.submesh([body_idx], append=True) if body_idx.size else None
    top_mesh = mesh.submesh([top_idx], append=True) if top_idx.size else None

    return SegmentMeshes(
        original=mesh,
        bottom=bottom_mesh,
        body=body_mesh,
        top=top_mesh,
        bottom_faces=bottom_idx,
        body_faces=body_idx,
        top_faces=top_idx,
    )



