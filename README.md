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

## TODO

* [ ] Add alignment for multiple channels
* [ ] Add colocalization for 3D case
* [ ] Add functional overview
* [ ] Add option to measure distance of spot from nuclear envelope
* [ ] Check cfg file and overwrite on changes?
* [ ] Create folders while using - not at the beginning
