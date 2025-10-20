"""Application entry point for the GTClay visual segmenter."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import numpy as np
from PySide6 import QtWidgets

from .mesh_processing import MeshProcessingError, SegmentMeshes, segment_mesh
from .widgets import MeshViewWidget, SidePanel


class MainWindow(QtWidgets.QMainWindow):
    """Main window hosting controls and 3D viewers."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GTClay - Segmentador de Tampas")
        self.resize(1400, 900)

        self.side_panel = SidePanel()
        self.side_panel.loadRequested.connect(self._handle_load_requested)
        self.side_panel.paintModeToggled.connect(self._toggle_manual_paint_mode)
        self.side_panel.clearPaintRequested.connect(self._clear_manual_paint)
        self.side_panel.applyPaintRequested.connect(self._apply_manual_paint)
        self.side_panel.autoToleranceChanged.connect(self._update_auto_tolerance)
        self.side_panel.autoPickRequested.connect(self._toggle_auto_pick)
        self.side_panel.setAutoToleranceValue(100)
        self.side_panel.setPaintingControlsEnabled(False)

        self.original_view = MeshViewWidget("Objeto Completo")
        self.top_view = MeshViewWidget("Tampa")
        self.body_view = MeshViewWidget("Corpo")
        self.bottom_view = MeshViewWidget("Base (Z=0)")

        self._segments: Optional[SegmentMeshes] = None
        self._manual_selected_faces: Set[int] = set()
        self._auto_pick_enabled: bool = False
        self._auto_tolerance_value: int = 100
        self.original_view.set_auto_pick_strength(self._auto_tolerance_value)
        self.side_panel.setAutoToleranceValue(self._auto_tolerance_value)

        central = QtWidgets.QWidget()
        outer_layout = QtWidgets.QHBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.side_panel.setFixedWidth(280)
        outer_layout.addWidget(self.side_panel)

        grid_container = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_container)
        grid_layout.setContentsMargins(6, 6, 6, 6)
        grid_layout.setSpacing(6)

        grid_layout.addWidget(self.original_view, 0, 0)
        grid_layout.addWidget(self.top_view, 0, 1)
        grid_layout.addWidget(self.body_view, 1, 0)
        grid_layout.addWidget(self.bottom_view, 1, 1)
        outer_layout.addWidget(grid_container, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Carregue um arquivo para iniciar.")

    def _display_segments(self, segments: SegmentMeshes) -> None:
        palette = {
            "original": (255, 179, 59, 255),
            "top": (235, 92, 70, 255),
            "body": (72, 176, 232, 255),
            "bottom": (64, 208, 138, 255),
        }
        self._segments = segments
        self._manual_selected_faces.clear()
        self.original_view.deactivate_manual_paint()
        self.original_view.cancel_auto_pick()
        self._auto_pick_enabled = False
        self.side_panel.setPaintModeChecked(False)
        self.side_panel.setAutoPickChecked(False)
        self.side_panel.updatePaintActionState(False)
        self.original_view.set_mesh(segments.original, palette["original"], shading="warm")
        self.top_view.set_mesh(segments.top, palette["top"], shading="warm")
        self.body_view.set_mesh(segments.body, palette["body"], shading="cool")
        self.bottom_view.set_mesh(segments.bottom, palette["bottom"], shading="cool")

    def _clear_views(self) -> None:
        self.original_view.deactivate_manual_paint()
        self.original_view.set_mesh(None, (0, 0, 0, 0))
        self.top_view.set_mesh(None, (0, 0, 0, 0))
        self.body_view.set_mesh(None, (0, 0, 0, 0))
        self.bottom_view.set_mesh(None, (0, 0, 0, 0))
        self._segments = None
        self._manual_selected_faces.clear()
        self.side_panel.setPaintingControlsEnabled(False)

    def _handle_load_requested(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Selecionar objeto 3D")
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(["Meshes (*.obj *.stl)", "OBJ (*.obj)", "STL (*.stl)", "Todos (*.*)"])
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = Path(selected[0])
        try:
            segments = segment_mesh(path)
        except MeshProcessingError as exc:
            QtWidgets.QMessageBox.critical(self, "Falha ao segmentar", str(exc))
            self.statusBar().showMessage("Erro ao processar o arquivo.")
            self._clear_views()
            return
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self,
                "Erro inesperado",
                f"Falha interna ao processar o arquivo:\n{exc}",
            )
            self.statusBar().showMessage("Erro ao processar o arquivo.")
            self._clear_views()
            return
        self._display_segments(segments)
        self.side_panel.setPaintingControlsEnabled(True)
        top_status = "detectada" if segments.top and not segments.top.is_empty else "nao detectada automaticamente"
        self.statusBar().showMessage(
            f"Tampa {top_status}. Faces: topo={segments.top.faces.shape[0] if segments.top else 0}, "
            f"corpo={segments.body.faces.shape[0] if segments.body else 0}, "
            f"base={segments.bottom.faces.shape[0] if segments.bottom else 0}."
        )

    def _toggle_manual_paint_mode(self, enabled: bool) -> None:
        if not self._segments:
            if enabled:
                QtWidgets.QMessageBox.information(
                    self,
                    "Nenhum objeto carregado",
                    "Carregue um objeto antes de usar o modo de pintura.",
                )
                self.side_panel.setPaintModeChecked(False)
            return
        if enabled:
            try:
                self.original_view.activate_manual_paint(
                    self._segments.original,
                    self._handle_manual_paint_update,
                    self._handle_auto_pick_result,
                )
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    self,
                    "Falha ao ativar pintura",
                    f"Nao foi possivel iniciar o modo de pintura:\n{exc}",
                )
                self.side_panel.setPaintModeChecked(False)
                self.side_panel.updatePaintActionState(False)
                return
            self.original_view.set_auto_pick_strength(self._auto_tolerance_value)
            self.statusBar().showMessage(
                "Modo pintura ativo. Esquerdo adiciona faces, direito remove. Clique em 'Aplicar tampa manual' para confirmar."
            )
            self.side_panel.updatePaintActionState(False)
        else:
            self._auto_pick_enabled = False
            self.side_panel.setAutoPickChecked(False)
            self.original_view.cancel_auto_pick()
            self.original_view.deactivate_manual_paint()
            self._manual_selected_faces.clear()
            self.side_panel.updatePaintActionState(False)
            self.statusBar().showMessage("Modo pintura desativado.")

    def _handle_manual_paint_update(self, selection: Set[int]) -> None:
        self._manual_selected_faces = set(selection)
        has_selection = bool(self._manual_selected_faces)
        self.side_panel.updatePaintActionState(has_selection)
        self.statusBar().showMessage(
            f"Faces selecionadas manualmente: {len(self._manual_selected_faces)}"
        )

    def _clear_manual_paint(self) -> None:
        if not self._segments:
            return
        self._auto_pick_enabled = False
        self.side_panel.setAutoPickChecked(False)
        self.original_view.cancel_auto_pick()
        self.original_view.clear_manual_paint()

    def _apply_manual_paint(self) -> None:
        if not self._segments:
            return
        if not self._manual_selected_faces:
            QtWidgets.QMessageBox.information(
                self,
                "Nenhuma selecao",
                "Selecione faces da borda antes de aplicar a tampa manual.",
            )
            return
        original = self._segments.original
        total_faces = len(original.faces)
        selected_idx = np.array(sorted(self._manual_selected_faces), dtype=int)
        bottom_idx = np.array(self._segments.bottom_faces, dtype=int)
        bottom_idx = bottom_idx[(bottom_idx >= 0) & (bottom_idx < total_faces)]
        mask = np.ones(total_faces, dtype=bool)
        if bottom_idx.size:
            mask[bottom_idx] = False
        mask[selected_idx] = False
        body_idx = np.where(mask)[0]

        bottom_mesh = (
            original.submesh([bottom_idx], append=True) if bottom_idx.size else None
        )
        top_mesh = original.submesh([selected_idx], append=True)
        body_mesh = original.submesh([body_idx], append=True) if body_idx.size else None

        new_segments = SegmentMeshes(
            original=original,
            bottom=bottom_mesh,
            body=body_mesh,
            top=top_mesh,
            bottom_faces=bottom_idx,
            body_faces=body_idx,
            top_faces=selected_idx,
        )
        self._display_segments(new_segments)
        self.side_panel.setPaintingControlsEnabled(True)
        self.side_panel.setPaintModeChecked(False)
        self.side_panel.updatePaintActionState(False)
        self.statusBar().showMessage(
            f"Tampa definida manualmente com {selected_idx.size} faces."
        )


    def _update_auto_tolerance(self, value: int) -> None:
        self._auto_tolerance_value = value
        self.original_view.set_auto_pick_strength(value)
        if self._auto_pick_enabled:
            self.original_view.prepare_auto_pick()
            self.statusBar().showMessage(
                "Filtro semi-automatico ajustado. Clique em uma face da tampa."
            )
    def _toggle_auto_pick(self, enabled: bool) -> None:
        if enabled:
            if not self._segments:
                QtWidgets.QMessageBox.information(
                    self,
                    "Nenhum objeto carregado",
                    "Carregue um objeto antes de usar a selecao semi-automatica.",
                )
                self.side_panel.setAutoPickChecked(False)
                return
            if not self.side_panel.paint_toggle.isChecked():
                self.side_panel.setPaintModeChecked(True)
                self._toggle_manual_paint_mode(True)
                if not self.side_panel.paint_toggle.isChecked():
                    self.side_panel.setAutoPickChecked(False)
                    return
            self._auto_pick_enabled = True
            self.original_view.set_auto_pick_strength(self._auto_tolerance_value)
            if self.original_view.prepare_auto_pick():
                self.statusBar().showMessage(
                    "Clique em uma face da tampa para a selecao semi-automatica."
                )
            else:
                self.statusBar().showMessage(
                    "Ative o modo pintura para usar a selecao semi-automatica."
                )
                self._auto_pick_enabled = False
                self.side_panel.setAutoPickChecked(False)
        else:
            self._auto_pick_enabled = False
            self.original_view.cancel_auto_pick()
            self.statusBar().showMessage("Selecao semi-automatica desativada.")

    def _handle_auto_pick_result(self, success: bool, added: int) -> None:
        if success:
            if added:
                msg = (
                    f"Selecao semi-automatica adicionou {added} faces. Total: {len(self._manual_selected_faces)}."
                )
            else:
                msg = "Selecao semi-automatica nao encontrou novas faces."
        else:
            msg = "Nao foi possivel identificar faces no clique. Ajuste o filtro e tente novamente."
        self.statusBar().showMessage(msg)
        if self._auto_pick_enabled:
            self.original_view.prepare_auto_pick()

def run() -> None:
    """Launch the Qt application."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


__all__ = ["MainWindow", "run"]















