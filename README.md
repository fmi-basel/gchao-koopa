# Koopa

Keenly optimized obliging picture analysis - Koopa is a luigi-pipeline based workflow to analyze cellular microscopy data of varying types - specializing on single particle analyses. The key features include:

* Preprocessing
  * For chromatic and camera alignment
  * Z-projection
  * Frame selection
* Spot methods
  * 2D/3D detection using `deepBlink` models
  * Tracking along time
  * Colocalization between channels
* Segmentation
  * Cellular segmentation (nuclei / cytoplasm) using `cellpose` models
  * Secondary segmentation (cytoplasm / nuclei) using mathematical filters
  * Other segmentation using mathematical filters or `segmentation_models` models

## Installation

Koopa can be installed using python's packaging index pypi - and has been tested for python versions 3.7-3.9.

```bash
pip install koopa
```

## Usage

Once installed, koopa can be accessed as a simple command line interface using two key commands:

```bash
# To create the config file which will have to be edited
koopa --create-config

# To run the actual workflow
koopa --config <path-to-config-file> --workers <number-of-workers>
```

## Functional Overview

Koopa is composed of a few number of luigi tasks that are connected.

* `SetupPipeline` - Initialized all folders and copies the configuration file.
* `ReferenceAlignment` - Creates a transformation matrix from the bead images.
* `Preprocess` - Opens, trims, aligns images depending on selection and saves them as tif.
* `Detect` - Raw spot detection on a single image/movie.
* `Track` - Linking of separated spots along time.
* `ColocalizeFrame` - Colocalize a pair of channels in a single time frame using linear sum assignment.
* `ColocalizeTrack` - Colocalize track pairs across a movie.
* `SegmentPrimary` - Segmentation of the main channel using cellpose (either on nuclei or cytoplasm).
* `SegmentSecondary` - Segmentation of the other channel (cytoplasm or nucleus respectively) using matematical filters.
* `SegmentOther` - Segmentation of additional features/channels using `segmentation_models` or mathematical filters.
* `Merge` - Combination of all tasks above (that have run on single images) into one summary file for downstream analyses.

## Potential TODOs

* [ ] Add alignment for multiple channels
* [ ] Add colocalization for 3D case
* [x] Add some integration tests
  * [ ] FISH 3D - with proper data once available
  * [x] FISH 2D - subset of specles
  * [x] Live cell - subset of smonster
* [x] Add functional overview
* [x] Add option to measure distance of spot from nuclear envelope
* [x] Check cfg file and overwrite on changes?
* [x] Create folders while using - not at the beginning
