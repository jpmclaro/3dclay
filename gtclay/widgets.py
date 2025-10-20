"""Custom Qt widgets for 3D visualization."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Set

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pyqtgraph.opengl import GLGridItem, GLLinePlotItem, GLMeshItem, GLViewWidget
import trimesh


class MeshViewWidget(GLViewWidget):
    """3D viewer with Fusion-like mouse controls."""

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self._setup_background()
        self.opts["distance"] = 600
        self.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.opts["elevation"] = 20
        self.opts["azimuth"] = 45
        self._mouse_button: Optional[QtCore.Qt.MouseButton] = None
        self._last_pos: Optional[QtCore.QPoint] = None
        self._mesh_item: Optional[GLMeshItem] = None
        self._current_vertices: Optional[np.ndarray] = None
        self._current_faces: Optional[np.ndarray] = None
        self._current_triangles: Optional[np.ndarray] = None
        self._base_colors: Optional[np.ndarray] = None
        self._paint_enabled = False
        self._paint_mesh: Optional[trimesh.Trimesh] = None
        self._paint_selected_faces: Set[int] = set()
        self._paint_callback: Optional[Callable[[Set[int]], None]] = None
        self._paint_brushing = False
        self._paint_button: Optional[QtCore.Qt.MouseButton] = None
        self._paint_face_neighbors: Optional[list[list[int]]] = None
        self._paint_brush_depth = 1
        self._paint_face_normals: Optional[np.ndarray] = None
        self._paint_face_centers: Optional[np.ndarray] = None
        self._auto_pick_pending = False
        self._auto_pick_strength = 1.0
        self._auto_pick_callback: Optional[Callable[[bool, int], None]] = None
        self._auto_pick_candidate = False
        self._auto_pick_start_pos: Optional[QtCore.QPoint] = None
        self._title_label = QtWidgets.QLabel(title, parent=self)
        self._title_label.setStyleSheet(
            "QLabel { color: #e0e0e0; background: rgba(0, 0, 0, 150); padding: 2px 6px; }"
        )
        self._axes = self._create_axes()
        for item in self._axes:
            self.addItem(item)
        self._grid = self._create_grid()
        self.addItem(self._grid)
        self._bounds = self._create_bounds()
        for edge in self._bounds:
            self.addItem(edge)

    def resizeEvent(self, evt: QtGui.QResizeEvent) -> None:  # noqa: D401
        super().resizeEvent(evt)
        if self._title_label:
            self._title_label.move(8, 8)

    def mousePressEvent(self, evt: QtGui.QMouseEvent) -> None:  # noqa: D401
        if (
            self._paint_enabled
            and self._auto_pick_pending
            and evt.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self._auto_pick_candidate = True
            self._auto_pick_start_pos = evt.position().toPoint()
            self._mouse_button = evt.button()
            self._last_pos = self._auto_pick_start_pos
            return

        if self._paint_enabled and evt.button() in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            self._auto_pick_candidate = False
            self._auto_pick_start_pos = None
            self._paint_brushing = True
            self._paint_button = evt.button()
            self._handle_paint_event(evt)
            return

        self._auto_pick_candidate = False
        self._auto_pick_start_pos = None
        self._mouse_button = evt.button()
        self._last_pos = evt.position().toPoint()
        super().mousePressEvent(evt)

    def mouseReleaseEvent(self, evt: QtGui.QMouseEvent) -> None:  # noqa: D401
        if (
            self._auto_pick_candidate
            and evt.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self._auto_pick_candidate = False
            self._auto_pick_start_pos = None
            self._mouse_button = None
            self._last_pos = None
            self._handle_auto_pick(evt)
            return

        if self._paint_enabled and self._paint_brushing:
            self._paint_brushing = False
            self._paint_button = None
            return

        super().mouseReleaseEvent(evt)
        self._mouse_button = None
        self._last_pos = None

    def mouseMoveEvent(self, evt: QtGui.QMouseEvent) -> None:  # noqa: D401
        if (
            self._auto_pick_candidate
            and evt.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            current = evt.position().toPoint()
            if (
                self._auto_pick_start_pos is not None
                and (current - self._auto_pick_start_pos).manhattanLength() > 6
            ):
                self._auto_pick_candidate = False
                self._auto_pick_start_pos = None
            self._last_pos = current

        if self._paint_enabled and self._paint_brushing:
            self._handle_paint_event(evt)
            return

        if self._last_pos is None:
            self._last_pos = evt.position().toPoint()
        delta = evt.position().toPoint() - self._last_pos
        self._last_pos = evt.position().toPoint()

        if self._mouse_button == QtCore.Qt.MouseButton.LeftButton:
            self.orbit(-delta.x() * 0.5, delta.y() * 0.5)
        elif self._mouse_button == QtCore.Qt.MouseButton.MiddleButton:
            self.pan(delta.x() * -0.5, delta.y() * 0.5, 0, relative="view")
        elif self._mouse_button == QtCore.Qt.MouseButton.RightButton:
            factor = 1 + (delta.y() * 0.01)
            self.opts["distance"] = max(1e-3, self.opts["distance"] * factor)
            self.update()
        else:
            super().mouseMoveEvent(evt)

    def wheelEvent(self, evt: QtGui.QWheelEvent) -> None:  # noqa: D401
        delta = evt.angleDelta().y()
        factor = 1 - delta / (8 * 120)
        self.opts["distance"] = max(1e-3, self.opts["distance"] * factor)
        self.update()

    def set_mesh(
        self,
        mesh: Optional["trimesh.Trimesh"],
        color: tuple[int, int, int, int],
        shading: str = "warm",
    ) -> None:
        """Display the provided mesh with the given color."""

        if self._mesh_item:
            self.removeItem(self._mesh_item)
            self._mesh_item = None
        if mesh is None:
            self._current_vertices = None
            self._current_faces = None
            self._current_triangles = None
            self._base_colors = None
            return
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        centroid = vertices.mean(axis=0)
        span = vertices.max(axis=0) - vertices.min(axis=0)
        distance = max(100.0, float(np.linalg.norm(span)) * 2.0)
        self.opts["center"] = QtGui.QVector3D(float(centroid[0]), float(centroid[1]), float(centroid[2]))
        self.opts["distance"] = distance
        self._current_vertices = vertices
        self._current_faces = faces
        self._current_triangles = vertices[faces]
        self._base_colors = self._shade_vertices(mesh, color, shading)
        colors = self._base_colors.copy()

        item = GLMeshItem(
            vertexes=vertices,
            faces=faces,
            vertexColors=colors,
            smooth=True,
            shader=None,
            glOptions="opaque",
        )
        self.addItem(item)
        self._mesh_item = item
        if self._paint_enabled and self._paint_selected_faces:
            self._update_paint_colors()

    def _shade_vertices(
        self,
        mesh: "trimesh.Trimesh",
        color: tuple[int, int, int, int],
        scheme: str,
    ) -> np.ndarray:
        base = np.array(color, dtype=np.float32) / 255.0
        if scheme == "cool":
            tint = np.array([0.82, 0.9, 1.08])
        else:
            tint = np.array([1.15, 1.05, 0.78])
        base_rgb = np.clip(base[:3] * tint, 0.0, 1.0)
        normals = mesh.vertex_normals
        light_dir = np.array([-0.4, -0.6, 0.7], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        diffuse = np.clip(normals.dot(light_dir), 0.0, 1.0).reshape(-1, 1)
        half_vec = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
        spec = np.clip(normals.dot(half_vec), 0.0, 1.0) ** 35
        ambient = 0.55
        diff_strength = 0.8
        color_rgb = base_rgb * (ambient + diff_strength * diffuse)
        specular_color = np.array([1.0, 0.92, 0.7])
        color_rgb += spec[:, None] * specular_color * 0.6
        color_rgb = np.clip(color_rgb, 0.0, 1.0)
        alpha = np.full((color_rgb.shape[0], 1), base[3])
        return np.hstack([color_rgb, alpha])

    # --- Manual paint helpers -------------------------------------------------

    def _update_paint_colors(self) -> None:
        if self._mesh_item is None or self._base_colors is None or self._current_vertices is None:
            return
        colors = self._base_colors.copy()
        if self._paint_selected_faces and self._current_faces is not None:
            face_indices = np.fromiter(self._paint_selected_faces, dtype=int)
            face_indices = face_indices[(face_indices >= 0) & (face_indices < self._current_faces.shape[0])]
            if face_indices.size:
                vertex_indices = np.unique(self._current_faces[face_indices].ravel())
                colors[vertex_indices] = np.array([0.95, 0.2, 0.2, 1.0], dtype=np.float32)
        self._mesh_item.setMeshData(
            vertexes=self._current_vertices,
            faces=self._current_faces,
            vertexColors=colors,
        )

    def activate_manual_paint(
        self,
        mesh: Optional["trimesh.Trimesh"],
        selection_callback: Optional[Callable[[Set[int]], None]],
        auto_pick_callback: Optional[Callable[[bool, int], None]] = None,
    ) -> None:
        if mesh is None or selection_callback is None:
            self.deactivate_manual_paint()
            return
        self._auto_pick_pending = False
        self._paint_mesh = mesh
        self._paint_face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        self._paint_face_centers = np.asarray(mesh.triangles_center, dtype=np.float32)
        adjacency: list[list[int]] = [[] for _ in range(len(mesh.faces))]
        if mesh.face_adjacency.size:
            for a, b in mesh.face_adjacency:
                adjacency[a].append(int(b))
                adjacency[b].append(int(a))
        self._paint_face_neighbors = adjacency
        self._paint_selected_faces.clear()
        self._paint_callback = selection_callback
        self._auto_pick_callback = auto_pick_callback
        self._paint_enabled = True
        self._paint_brushing = False
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self._update_paint_colors()

    def deactivate_manual_paint(self) -> None:
        self._paint_enabled = False
        self._paint_mesh = None
        self._paint_face_normals = None
        self._paint_face_centers = None
        self._paint_callback = None
        self._auto_pick_callback = None
        self._paint_selected_faces.clear()
        self._paint_face_neighbors = None
        self._paint_brushing = False
        self._paint_button = None
        self._auto_pick_pending = False
        self.unsetCursor()
        if self._mesh_item is not None and self._base_colors is not None and self._current_vertices is not None:
            self._mesh_item.setMeshData(
                vertexes=self._current_vertices,
                faces=self._current_faces,
                vertexColors=self._base_colors,
            )

    def clear_manual_paint(self) -> None:
        if not self._paint_selected_faces:
            return
        self._paint_selected_faces.clear()
        self._update_paint_colors()
        if self._paint_callback:
            self._paint_callback(set())

    def manual_paint_selection(self) -> Set[int]:
        return set(self._paint_selected_faces)

    def set_auto_pick_strength(self, slider_value: int) -> None:
        self._auto_pick_strength = max(0.1, slider_value / 100.0)

    def prepare_auto_pick(self) -> bool:
        if not self._paint_enabled:
            return False
        self._auto_pick_pending = True
        self._paint_brushing = False
        self._paint_button = None
        return True

    def cancel_auto_pick(self) -> None:
        self._auto_pick_pending = False

    def _handle_auto_pick(self, evt: QtGui.QMouseEvent) -> None:
        self._auto_pick_pending = False
        if self._paint_mesh is None:
            if self._auto_pick_callback:
                self._auto_pick_callback(False, 0)
            return
        pos = evt.position() if hasattr(evt, "position") else evt.localPos()
        ray = self._compute_mouse_ray(pos)
        if ray is None:
            if self._auto_pick_callback:
                self._auto_pick_callback(False, 0)
            return
        origin, direction = ray
        face = self._pick_face(origin, direction)
        if face is None:
            if self._auto_pick_callback:
                self._auto_pick_callback(False, 0)
            return
        faces = self._auto_select_faces(face)
        if not faces:
            if self._auto_pick_callback:
                self._auto_pick_callback(False, 0)
            return
        previous = set(self._paint_selected_faces)
        self._paint_selected_faces.update(faces)
        added_faces = self._paint_selected_faces - previous
        if added_faces:
            self._update_paint_colors()
            if self._paint_callback:
                self._paint_callback(set(self._paint_selected_faces))
        if self._auto_pick_callback:
            self._auto_pick_callback(True, len(added_faces))
    def _auto_select_faces(self, seed_face: int) -> Set[int]:
        if (
            self._paint_face_neighbors is None
            or self._paint_face_normals is None
            or self._paint_face_centers is None
        ):
            return {seed_face}
        normals = self._paint_face_normals
        centers = self._paint_face_centers
        base_normal = normals[seed_face]
        norm = np.linalg.norm(base_normal)
        if norm < 1e-9:
            return {seed_face}
        plane_normal = base_normal / norm
        plane_origin = centers[seed_face]
        factor = self._auto_pick_strength
        max_angle = np.deg2rad(6.0 * factor + 2.0)
        cos_threshold = np.cos(max_angle)
        dist_tolerance = 0.4 * factor + 0.1
        visited: Set[int] = {seed_face}
        queue: deque[int] = deque([seed_face])
        while queue:
            current = queue.popleft()
            for nb in self._paint_face_neighbors[current]:
                if nb in visited:
                    continue
                n = normals[nb]
                if np.dot(n, plane_normal) < cos_threshold:
                    continue
                distance = abs(np.dot(centers[nb] - plane_origin, plane_normal))
                if distance > dist_tolerance:
                    continue
                visited.add(nb)
                queue.append(nb)
        return visited

    def _handle_paint_event(self, evt: QtGui.QMouseEvent) -> None:
        if not self._paint_enabled or self._paint_mesh is None:
            return
        pos = evt.position() if hasattr(evt, "position") else evt.localPos()
        ray = self._compute_mouse_ray(pos)
        if ray is None:
            return
        origin, direction = ray
        face = self._pick_face(origin, direction)
        if face is None:
            return
        faces_to_update = self._collect_brush_faces(face)
        if not faces_to_update:
            return
        if self._paint_button == QtCore.Qt.MouseButton.RightButton:
            changed = bool(faces_to_update & self._paint_selected_faces)
            self._paint_selected_faces.difference_update(faces_to_update)
        else:
            changed = bool(faces_to_update - self._paint_selected_faces)
            self._paint_selected_faces.update(faces_to_update)
        if not changed:
            return
        self._update_paint_colors()
        if self._paint_callback:
            self._paint_callback(set(self._paint_selected_faces))

    def _collect_brush_faces(self, start_face: int) -> Set[int]:
        if self._paint_face_neighbors is None:
            return {start_face}
        visited: Set[int] = {start_face}
        frontier: Set[int] = {start_face}
        for _ in range(self._paint_brush_depth):
            next_frontier: Set[int] = set()
            for face in frontier:
                for nb in self._paint_face_neighbors[face]:
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            if not next_frontier:
                break
            frontier = next_frontier
        return visited

    def _compute_mouse_ray(self, pos: QtCore.QPointF) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self._current_vertices is None:
            return None
        w = max(1, self.width())
        h = max(1, self.height())
        ndc_x = (2.0 * pos.x() / w) - 1.0
        ndc_y = 1.0 - (2.0 * pos.y() / h)
        near = QtGui.QVector4D(ndc_x, ndc_y, -1.0, 1.0)
        far = QtGui.QVector4D(ndc_x, ndc_y, 1.0, 1.0)
        proj = self.projectionMatrix()
        view = self.viewMatrix()
        combined = proj * view
        inverted, success = combined.inverted()
        if not success:
            return None
        near_world = inverted.map(near)
        far_world = inverted.map(far)
        near_np = np.array([near_world.x(), near_world.y(), near_world.z(), near_world.w()], dtype=float)
        far_np = np.array([far_world.x(), far_world.y(), far_world.z(), far_world.w()], dtype=float)
        if abs(near_np[3]) > 1e-8:
            near_np[:3] /= near_np[3]
        if abs(far_np[3]) > 1e-8:
            far_np[:3] /= far_np[3]
        origin = near_np[:3]
        direction = far_np[:3] - origin
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        direction /= norm
        return origin, direction

    def _pick_face(self, origin: np.ndarray, direction: np.ndarray) -> Optional[int]:
        if self._current_triangles is None or self._current_faces is None:
            return None
        triangles = self._current_triangles
        v0 = triangles[:, 0]
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]
        e1 = v1 - v0
        e2 = v2 - v0
        pvec = np.cross(direction, e2)
        det = (e1 * pvec).sum(axis=1)
        mask = np.abs(det) > 1e-12
        if not np.any(mask):
            return None
        inv_det = np.zeros_like(det)
        inv_det[mask] = 1.0 / det[mask]
        tvec = origin - v0
        u = (tvec * pvec).sum(axis=1) * inv_det
        mask &= (u >= 0.0) & (u <= 1.0)
        if not np.any(mask):
            return None
        qvec = np.cross(tvec, e1)
        v = (direction * qvec).sum(axis=1) * inv_det
        mask &= (v >= 0.0) & (u + v <= 1.0)
        if not np.any(mask):
            return None
        t = (e2 * qvec).sum(axis=1) * inv_det
        mask &= t > 1e-6
        if not np.any(mask):
            return None
        t_valid = np.where(mask, t, np.inf)
        face_idx = int(np.argmin(t_valid))
        if not np.isfinite(t_valid[face_idx]):
            return None
        return face_idx

    def _setup_background(self) -> None:
        self.setBackgroundColor(QtGui.QColor(200, 224, 252))

    def _create_axes(self) -> list[GLLinePlotItem]:
        """Return RGB axes using line items for custom coloring."""

        axis_lines = [
            ((0.0, 0.0, 0.0), (120.0, 0.0, 0.0), (0.92, 0.34, 0.34, 1.0)),
            ((0.0, 0.0, 0.0), (0.0, 120.0, 0.0), (0.27, 0.78, 0.37, 1.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 120.0), (0.20, 0.36, 0.86, 1.0)),
        ]
        axes: list[GLLinePlotItem] = []
        for start, end, color in axis_lines:
            line = GLLinePlotItem(
                pos=np.array([start, end], dtype=np.float32),
                color=color,
                width=3,
                mode="lines",
                antialias=True,
            )
            axes.append(line)
        return axes

    def _create_grid(self) -> GLGridItem:
        grid = GLGridItem()
        grid.setSize(x=400, y=400, z=1)
        grid.setSpacing(x=10, y=10, z=1)
        grid.translate(0, 0, 0)
        grid.setColor(QtGui.QColor(150, 150, 150, 180))
        return grid

    def _create_bounds(self) -> list[GLLinePlotItem]:
        """Create a translucent bounding box similar to CAD view cube."""

        half = 200.0
        corners = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
                [-half, -half, half * 2],
                [half, -half, half * 2],
                [half, half, half * 2],
                [-half, half, half * 2],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        color = (0.7, 0.78, 0.88, 0.5)
        items: list[GLLinePlotItem] = []
        for start, end in edges:
            pts = np.array([corners[start], corners[end]], dtype=np.float32)
            line = GLLinePlotItem(pos=pts, color=color, mode="lines", antialias=True, width=1)
            items.append(line)
        return items


class SidePanel(QtWidgets.QWidget):
    """Left panel for controls."""

    loadRequested = QtCore.Signal()
    paintModeToggled = QtCore.Signal(bool)
    clearPaintRequested = QtCore.Signal()
    applyPaintRequested = QtCore.Signal()
    autoToleranceChanged = QtCore.Signal(int)
    autoPickRequested = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel("GTClay Segmenter")
        heading.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; }")
        layout.addWidget(heading)

        description = QtWidgets.QLabel(
            "Carregue um objeto 3D (.obj ou .stl) para detectar tampa, corpo e base."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.load_button = QtWidgets.QPushButton("Carregar objeto...")
        self.load_button.clicked.connect(self.loadRequested.emit)
        layout.addWidget(self.load_button)

        self.paint_toggle = QtWidgets.QPushButton("Modo pintura da tampa")
        self.paint_toggle.setCheckable(True)
        self.paint_toggle.setEnabled(False)
        self.paint_toggle.toggled.connect(self.paintModeToggled.emit)
        layout.addWidget(self.paint_toggle)

        paint_hint = QtWidgets.QLabel(
            "Use o botao esquerdo do mouse para adicionar faces e o direito para remover."
        )
        paint_hint.setWordWrap(True)
        paint_hint.setStyleSheet("color: #555; font-size: 12px;")
        layout.addWidget(paint_hint)

        self.auto_label = QtWidgets.QLabel("Filtro semi-automatico: 1.00x")
        layout.addWidget(self.auto_label)

        self.auto_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.auto_slider.setRange(10, 500)
        self.auto_slider.setValue(100)
        self.auto_slider.setEnabled(False)
        self.auto_slider.valueChanged.connect(self._handle_auto_slider)
        layout.addWidget(self.auto_slider)

        self.auto_pick_button = QtWidgets.QPushButton("Selecionar faces automaticamente")
        self.auto_pick_button.setCheckable(True)
        self.auto_pick_button.setEnabled(False)
        self.auto_pick_button.toggled.connect(self.autoPickRequested.emit)
        layout.addWidget(self.auto_pick_button)

        self.apply_paint_button = QtWidgets.QPushButton("Aplicar tampa manual")
        self.apply_paint_button.setEnabled(False)
        self.apply_paint_button.clicked.connect(self.applyPaintRequested.emit)
        layout.addWidget(self.apply_paint_button)

        self.clear_paint_button = QtWidgets.QPushButton("Limpar pintura")
        self.clear_paint_button.setEnabled(False)
        self.clear_paint_button.clicked.connect(self.clearPaintRequested.emit)
        layout.addWidget(self.clear_paint_button)

        layout.addStretch(1)

    def setPaintingControlsEnabled(self, enabled: bool) -> None:
        self.paint_toggle.setEnabled(enabled)
        if not enabled:
            self.setPaintModeChecked(False)
            self.setAutoPickChecked(False)
        self.apply_paint_button.setEnabled(False)
        self.clear_paint_button.setEnabled(False)
        self.auto_pick_button.setEnabled(enabled)
        self.auto_slider.setEnabled(enabled)

    def setPaintModeChecked(self, checked: bool) -> None:
        self.paint_toggle.blockSignals(True)
        self.paint_toggle.setChecked(checked)
        self.paint_toggle.blockSignals(False)

    def setAutoPickChecked(self, checked: bool) -> None:
        self.auto_pick_button.blockSignals(True)
        self.auto_pick_button.setChecked(checked)
        self.auto_pick_button.blockSignals(False)

    def updatePaintActionState(self, has_selection: bool) -> None:
        self.apply_paint_button.setEnabled(has_selection)
        self.clear_paint_button.setEnabled(has_selection)

    def setAutoToleranceValue(self, value: int) -> None:
        self.auto_slider.blockSignals(True)
        self.auto_slider.setValue(value)
        self.auto_slider.blockSignals(False)
        self._update_auto_label(value / 100.0)

    def _handle_auto_slider(self, value: int) -> None:
        self._update_auto_label(value / 100.0)
        self.autoToleranceChanged.emit(value)

    def _update_auto_label(self, factor: float) -> None:
        self.auto_label.setText(f"Filtro semi-automatico: {factor:.2f}x")








