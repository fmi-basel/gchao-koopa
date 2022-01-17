import glob
import os
import re
from typing import List

from tqdm import tqdm
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pystackreg
import skimage.io

from config import CustomConfig


def create_directories(basedir: str) -> None:
    """Create all analysis directories for a given path."""
    directories = [
        "preprocessed",
        "segmentation",
        "detection_ms2",
        "detection_suntag",
        "track_ms2",
        "track_suntag",
        "colocalization",
    ]
    for folder in directories:
        path = os.path.join(basedir, folder)
        if not os.path.exists(path):
            os.makedirs(path)


def parse_nd(filename: str) -> dict:
    """Parse .nd configuration files as dictionary."""
    nd_data = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            try:
                key, value = re.search(r'^"(.+)", "?([^"]+)"?\s$', line).groups()
                nd_data[key] = value
            except AttributeError:
                pass
    return nd_data


def open_nd_file(fname_nd: str) -> np.ndarray:
    """Read and merge all files mentioned in one nd file."""
    nd_data = parse_nd(fname_nd)
    basename = os.path.splitext(fname_nd)[0]

    # Parse channels
    channels = int(nd_data["NWavelengths"])
    images = []

    for channel in range(1, channels + 1):
        channel_name = nd_data[f"WaveName{channel}"]
        fname_image = f"{basename}_w{channel}{channel_name}.stk"
        image = skimage.io.imread(fname_image)
        images.append(image)

    # Merge channels
    try:
        image = np.stack(images, axis=0)
    except ValueError:
        raise ValueError(f"Could not merge channels. Check shapes for {fname_nd}.")
    return image


class ReferenceAlignment(luigi.Task):
    def output(self):
        return [
            luigi.LocalTarget(
                os.path.join(CustomConfig().analysis_dir, "alignment.npy")
            ),
            luigi.LocalTarget(
                os.path.join(CustomConfig().analysis_dir, "alignment.png"),
            ),
        ]

    def run(self):
        self.load_images()
        self.register_alignment_matrix()
        self.save_alignment()
        self.plot_alignment()

    @staticmethod
    def read_stack(fnames: List[os.PathLike]) -> np.ndarray:
        """Maximum project a stack of image files."""
        images = [skimage.io.imread(f) for f in fnames]
        return np.max(images, axis=0)

    def load_images(self) -> None:
        """Load reference and alignment images."""
        fnames_reference = sorted(
            glob.glob(
                os.path.join(
                    CustomConfig().alignment_dir,
                    f"*{CustomConfig().channel_reference}.tif",
                )
            )
        )
        fnames_transform = sorted(
            glob.glob(
                os.path.join(
                    CustomConfig().alignment_dir,
                    f"*{CustomConfig().channel_alignment}.tif",
                )
            )
        )
        self.image_reference = self.read_stack(fnames_reference)
        self.image_transform = self.read_stack(fnames_transform)

    def register_alignment_matrix(self) -> None:
        """Calculate a rigid body transformation matrix."""
        sr = pystackreg.StackReg(pystackreg.StackReg.RIGID_BODY)
        sr.register(self.image_reference, self.image_transform)
        self.sr = sr

    def save_alignment(self) -> None:
        """Save alignment matrix to file."""
        np.save(self.output()[0].path, self.sr.get_matrix())

    def plot_alignment(self) -> None:
        """Plot chromatic transform before and after alignment."""
        transform = self.sr.transform(self.image_transform)
        _, ax = plt.subplots(1, 2, figsize=(20, 20))
        ax[0].set_title("Pre-alignment")
        ax[0].imshow(self.image_reference, cmap="Greens")
        ax[0].imshow(self.image_transform, cmap="Reds", alpha=0.5)
        ax[1].set_title("Post-alignment")
        ax[1].imshow(self.image_reference, cmap="Greens")
        ax[1].imshow(transform, cmap="Reds", alpha=0.5)
        plt.savefig(self.output()[1].path)
        plt.close()


class Preprocess(luigi.Task):
    """Task to open, trim, and align images."""

    FileID = luigi.Parameter()

    @property
    def input_name(self):
        input_name = os.path.join(CustomConfig().image_dir, f"{self.FileID}.nd")
        if not os.path.exists(input_name):
            raise ValueError(f"File {input_name} does not exist.")
        return input_name

    def requires(self):
        return ReferenceAlignment()

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                CustomConfig().analysis_dir, "preprocessed", f"{self.FileID}.tif"
            )
        )

    def run(self):
        create_directories(basedir=CustomConfig().analysis_dir)
        image = open_nd_file(self.input_name)
        image = image[:, CustomConfig().frame_start : CustomConfig().frame_end]

        self.load_alignment()
        image = self.align_stack(image)
        skimage.io.imsave(self.output().path, image, check_contrast=False)

    def load_alignment(self):
        """Load alignment matrix from file."""
        sr = pystackreg.StackReg(pystackreg.StackReg.RIGID_BODY)
        matrix = np.load(self.requires().output()[0].path)
        sr.set_matrix(matrix)
        self.sr = sr

    def align_stack(self, stack: np.ndarray) -> np.ndarray:
        """Align a stack of images (two channels only)."""
        transform_index = sorted(
            [CustomConfig().channel_alignment, CustomConfig().channel_reference]
        ).index(CustomConfig().channel_alignment)

        stack[transform_index] = np.array(
            [
                self.sr.transform(i)
                for i in tqdm(stack[transform_index], desc="Align frame")
            ],
            dtype=np.uint16,
        )
        return stack
