"""Setup file for pypi package called koopa."""

# python setup.py sdist
# twine upload dist/latest-version.tar.gz

import textwrap
from setuptools import find_packages
from setuptools import setup

setup(
    # Description
    name="koopa",
    version="0.0.10",
    license="MIT",
    description="Workflow for analysis of cellular microscopy data.",
    long_description_content_type="text/plain",
    long_description=textwrap.dedent(
        """Keenly optimized obliging picture analysis. Koopa is the core package
for a prefect-pipeline based workflow to analyze cellular microscopy data
of varying types - specializing on single particle analyses."""
    ),
    # Installation
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "czifile==2019.7.2",
        "numba==0.56.4",
        "numpy==1.23.5",
        "pandas==1.3.5",
        "pyarrow==6.0.1",
        "pystackreg==0.2.5",
        "scikit-image==0.19.3",
        "scipy==1.4.1",
        "tifffile==2020.09.03",
        "trackpy==0.5.0",
    ],
    extras_require=dict(
        deepblink=["tensorflow==2.7", "deepblink"],
        cellpose=["torch==1.10.1", "cellpose"],
        segmodels=["segmentation_models==1.0.1"],
    ),
    # Metadata
    author="Bastian Eichenberger",
    author_email="bastian@eichenbergers.ch",
    url="https://github.com/bbquercus/koopa/",
    project_urls={
        "Documentation": "https://github.com/BBQuercus/koopa/wiki",
        "Changelog": "https://github.com/BBQuercus/koopa/releases",
        "Issue Tracker": "https://github.com/bbquercus/koopa/issues",
    },
    keywords=["biomedical", "bioinformatics", "image analysis"],
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Utilities",
    ],
)
