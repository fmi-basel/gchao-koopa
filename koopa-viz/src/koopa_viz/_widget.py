import configparser
import glob
import os

from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import napari
import numpy as np
import pandas as pd
import tifffile


class KoopaWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        # General config
        self.viewer = napari_viewer
        self.spots_cols = ["frame", "y", "x"]
        self.track_cols = ["particle", "frame", "y", "x"]
        self.params = [
            "blending",
            "color_mode",
            "colormap",
            "contour",
            "contrast_limits",
            "face_color",
            "gamma",
            "opacity",
            "out_of_slice_display",
            "size",
            "symbol",
            "visible",
        ]

        # Viewer model params - https://napari.org/stable/api/napari.Viewer
        self.image_params = dict(blending="additive")
        self.label_params = dict(
            blending="translucent", num_colors=50, opacity=0.7
        )
        self.point_params = dict(
            edge_color="black",
            face_color="white",
            size=5,
            out_of_slice_display=True,
        )
        self.track_params = dict(tail_width=8, tail_length=30, head_length=2)

        # Build plugin layout
        self.setLayout(QVBoxLayout())
        self.setup_logo_header()
        self.setup_config_parser()
        self.setup_file_dropdown()
        self.setup_file_navigation()
        self.setup_viewing_options()
        self.setup_progress_bar()

    def clear_viewer(self):
        """Reset all layers to an empty window."""
        self.viewer.reset_view()
        self.viewer.layers.clear()

    def setup_logo_header(self):
        """Prepare widget for header."""
        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(QLabel("<h1>Koopa</h1>"))
        widget.layout().addWidget(
            QLabel("Keenly optimized obliging picture analysis.")
        )
        self.layout().addWidget(widget)

    def setup_config_parser(self):
        """Prepare widget for config reader."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Analysis Directory:</b>"))

        btn_widget = QPushButton("Select")
        btn_widget.clicked.connect(self.open_file_dialog)
        widget.layout().addWidget(btn_widget)
        self.layout().addWidget(widget)

    def setup_file_dropdown(self):
        """Prepare widget for single file selection."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Current File:</b>"))

        self.dropdown_widget = QComboBox()
        widget.layout().addWidget(self.dropdown_widget)

        btn_widget = QPushButton("Load")
        btn_widget.clicked.connect(self.load_file)
        widget.layout().addWidget(btn_widget)

        self.file_dropdown = widget
        self.file_dropdown.setDisabled(True)
        self.layout().addWidget(self.file_dropdown)

    def setup_file_navigation(self):
        """Prepare widget for file navigation."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Navigate Files:</b>"))
        prev_widget = QPushButton("Previous Image")
        prev_widget.clicked.connect(lambda: self.change_file("prev"))
        widget.layout().addWidget(prev_widget)
        next_widget = QPushButton("Next Image")
        next_widget.clicked.connect(lambda: self.change_file("next"))
        widget.layout().addWidget(next_widget)

        self.file_navigation = widget
        self.file_navigation.setDisabled(True)
        self.layout().addWidget(self.file_navigation)

    def setup_viewing_options(self):
        """Prepare widget accessibility options."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Viewing Options:</b>"))

        hideall_widget = QPushButton("Hide All Layers")
        hideall_widget.clicked.connect(self.hide_layers)
        widget.layout().addWidget(hideall_widget)

        settings_save_widget = QPushButton("Save Settings")
        settings_save_widget.clicked.connect(self.save_settings)
        widget.layout().addWidget(settings_save_widget)
        settings_apply_widget = QPushButton("Apply Settings")
        settings_apply_widget.clicked.connect(self.apply_settings)
        widget.layout().addWidget(settings_apply_widget)
        self.layout().addWidget(widget)

    def setup_progress_bar(self):
        """Prepare widget with data loading progress bar."""
        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.layout().addWidget(self.pbar)

    def open_file_dialog(self):
        """Dialog to select analysis directory."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.DirectoryOnly)
        if dialog.exec_():
            self.clear_viewer()
            self.analysis_path = dialog.selectedFiles()[0]
            self.run_config_parser()
            self.get_file_list()

    def run_config_parser(self):
        """Retrieve config file from analysis directory."""
        config_file = os.path.join(
            os.path.abspath(self.analysis_path), "koopa.cfg"
        )
        if not os.path.exists(config_file):
            napari.utils.notifications.show_error(
                "Koopa config file does not exist!"
            )
            return None

        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.do_timeseries = eval(self.config["General"]["do_timeseries"])
        self.do_3d = eval(self.config["General"]["do_3d"])

    def load_file(self):
        """Open all associated files and enable editing."""
        self.name = self.dropdown_widget.currentText()

        # Canvas reset
        self.pbar.setValue(0)
        self.clear_viewer()
        self.pbar.setValue(10)

        # Images
        self.load_image()
        self.pbar.setValue(25)

        # Segmaps
        self.load_segmentation_cells()
        self.pbar.setValue(50)
        if eval(self.config["SegmentationOther"]["sego_enabled"]):
            self.load_segmentation_other()
        self.pbar.setValue(75)

        # Points
        self.load_detection_raw()
        self.pbar.setValue(85)
        if eval(self.config["SpotsColocalization"]["coloc_enabled"]):
            self.load_colocalization()
        self.pbar.setValue(100)

    def change_file(self, option: str):
        """Navigation prev/next to change files faster."""
        file_idx = self.files.index(self.dropdown_widget.currentText())
        if file_idx == 0 and option == "prev":
            raise ValueError("Already at the beginning!")
        if file_idx == len(self.files) and option == "next":
            raise ValueError("Already at the end!")

        new_idx = file_idx + 1 if option == "next" else file_idx - 1
        self.dropdown_widget.setCurrentText(self.files[new_idx])
        self.load_file()

    def get_file_list(self):
        """Find all files in analysis directory and update dropdown."""
        files = sorted(
            glob.glob(
                os.path.join(self.analysis_path, "preprocessed", "*.tif")
            )
        )
        files = sorted(
            [os.path.basename(f).replace(".tif", "") for f in files]
        )
        self.files = sorted(
            [f[:-17] for f in files]
        )

        self.file_dropdown.setDisabled(False)
        self.file_navigation.setDisabled(False)
        self.dropdown_widget.clear()
        self.dropdown_widget.addItems(self.files)

    def load_image(self):
        """Open and display raw image data."""
        fname = glob.glob(os.path.join(
            self.analysis_path, "preprocessed", f"{self.name}-*.tif"
        ))[0]
        self.image = tifffile.imread(fname)
        for idx, channel in enumerate(self.image):
            self.viewer.add_image(
                channel, name=f"Channel {idx}", **self.image_params
            )

    def load_segmentation_cells(self):
        """Open and display nuclear/cytoplasmic segmentation maps."""
        for name in ["nuclei", "cyto"]:
            fname = glob.glob(os.path.join(
                self.analysis_path, f"segmentation_{name}", f"{self.name}-*.tif"
            ))[0]
            if not os.path.exists(fname):
                continue
            segmap = tifffile.imread(fname).astype(int)
            self.viewer.add_labels(
                segmap,
                name=f"Segmentation {name.capitalize()}",
                **self.label_params,
            )

    def load_segmentation_other(self):
        """Open and display additional segmentation maps."""
        for channel in eval(self.config["SegmentationOther"]["sego_channels"]):
            fname = glob.glob(os.path.join(
                self.analysis_path,
                f"segmentation_{channel}",
                f"{self.name}-*.tif",
            ))[0]
            segmap = tifffile.imread(fname).astype(int)
            self.viewer.add_labels(
                segmap, name=f"Segmentation C{channel}", **self.label_params
            )

    def load_detection_raw(self):
        """Open and display raw spot detection points."""

        for channel in eval(self.config["SpotsDetection"]["detect_channels"]):
            folder = (
                f"detection_final_c{channel}"
                if self.do_3d or self.do_timeseries
                else f"detection_raw_c{channel}"
            )
            fname = glob.glob(os.path.join(
                self.analysis_path, folder, f"{self.name}-*.parq"
            ))[0]
            df = pd.read_parquet(fname)

            if self.do_timeseries:
                self.viewer.add_tracks(
                    df[self.track_cols],
                    name=f"Track C{channel}",
                    **self.track_params,
                )
            else:
                self.viewer.add_points(
                    df[self.spots_cols],
                    name=f"Detection C{channel}",
                    **self.point_params,
                )

    def load_colocalization(self):
        """Open and display colocalization pairs (colocalized vs. non)."""

        for i, j in eval(self.config["SpotsColocalization"]["coloc_channels"]):
            fname = glob.glob(os.path.join(
                self.analysis_path,
                f"colocalization_{i}-{j}",
                f"{self.name}-*.parq",
            ))[0]
            df = pd.read_parquet(fname)
            df_coloc = df[df["channel"] == i]
            df_empty = df[df["channel"] == j]

            if self.do_timeseries:
                df_coloc = df_coloc.loc[
                    df_coloc[f"coloc_particle_{i}-{j}"].isna(), self.track_cols
                ]
                df_empty = df_empty.loc[
                    ~df_empty[f"coloc_particle_{i}-{j}"].isna(),
                    self.track_cols,
                ]
                self.viewer.add_tracks(
                    df_coloc,
                    name=f"Track {i}-{j} Coloc",
                    colormap="red",
                    **self.track_params,
                )
                self.viewer.add_tracks(
                    df_empty,
                    name=f"Track {i}-{j} Empty",
                    colormap="blue",
                    **self.track_params,
                )
            else:
                df_coloc = df_coloc.loc[
                    df_coloc[f"coloc_particle_{i}-{j}"] != 0, self.spots_cols
                ]
                df_empty = df_empty.loc[
                    df_empty[f"coloc_particle_{i}-{j}"] == 0, self.spots_cols
                ]
                bland_point_params = self.point_params.copy()
                bland_point_params.pop("face_color")
                self.viewer.add_points(
                    df_coloc,
                    name=f"Detection {i}-{j} Coloc",
                    face_color="red",
                    **bland_point_params,
                )
                self.viewer.add_points(
                    df_empty,
                    name=f"Detection {i}-{j} Empty",
                    face_color="blue",
                    **bland_point_params,
                )

    def hide_layers(self):
        """Set visibility of all layers to False."""
        for layer in self.viewer.layers:
            layer.visible = False

    @staticmethod
    def rgb_to_hex(rgb: np.ndarray):
        """Convert (R, G, B) format to #RRGGBB."""
        return ("#{:X}{:X}{:X}").format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    def save_settings(self):
        """Save user specific viewing settings (e.g. contrast)."""
        self.settings = {}
        for idx, layer in enumerate(self.viewer.layers):
            layer_settings = {}
            for param in self.params:
                if hasattr(layer, param):
                    # NOTE currently doesn't get set properly via interface
                    # Bug in napari not in plugin
                    if param == "face_color":
                        value = self.rgb_to_hex(layer.face_color[0, :3])
                    elif param == "size":
                        value = layer.size[0, 0]
                    else:
                        value = getattr(layer, param)
                    layer_settings[param] = value
            self.settings[idx] = layer_settings
        napari.utils.notifications.show_info("Current settings saved.")

    def apply_settings(self):
        """Apply previously saved user specific viewing settings."""
        if not hasattr(self, "settings"):
            napari.utils.notifications.show_error("No settings saved!")
            return None

        for idx, layer in enumerate(self.viewer.layers):
            if idx in self.settings:
                for param, value in self.settings[idx].items():
                    setattr(layer, param, value)
        napari.utils.notifications.show_info("Saved settings applied.")
