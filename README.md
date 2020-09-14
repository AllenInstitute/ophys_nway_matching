[![CircleCI](https://circleci.com/gh/AllenInstitute/ophys_nway_matching.svg?style=svg)](https://circleci.com/gh/AllenInstitute/ophys_nway_matching)
[![codecov](https://codecov.io/gh/AllenInstitute/ophys_nway_matching/branch/master/graph/badge.svg?token=y5Nt5RnMwB)](https://codecov.io/gh/AllenInstitute/ophys_nway_matching)

# ophys_nway_matching
N-way matching of segmented cell ROIs

# Docker and Singularity
A docker image is built in CircleCI and pushed to [dockerhub](https://hub.docker.com/repository/docker/alleninstitutepika/ophys_nway_matching) tagged as either `master` or `develop`.

Singularity should be able to run this docker image directly:
```
singularity run docker://alleninstitutepika/ophys_nway_matching:develop python -m pytest /ophys_nway_matching
```
or
```
singularity run docker://alleninstitutepika/ophys_nway_matching:develop python -m nway.nway_matching --help
```

It appears the calling singularity in this way intelligently uses the local caches for both docker and singularity. There is an overhead for downloading an updated docker image and translating it to a singularity image. That cost is incurred only when the docker image has changed.

# quick start

```
conda create -n nwaytest python=3.6.4
conda activate nwaytest
pip install git+https://github.com/AllenInstitute/ophys_nway_matching
python -m nway.nway_matching --input_json tmp_example/input.json --output_json tmp_example/output.json
```

This creates `tmp_example/output.json`. The intent is that this file contains everything you need to know, both results and diagnostic metrics.

Some basic visualizations of the results are in `nway.diagnostics`
```
python -m nway.diagnostics --input_json tmp_example/output.json --output_pdf tmp_example/output.pdf

```

This should create `tmp_example/output.pdf`.

# Level of support
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support, as it is under active development. The community is welcome to submit issues, but you should not expect an active response.
