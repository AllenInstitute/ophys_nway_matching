[![codecov](https://codecov.io/gh/AllenInstitute/ophys_nway_matching/branch/master/graph/badge.svg?token=y5Nt5RnMwB)](https://codecov.io/gh/AllenInstitute/ophys_nway_matching)

# ophys_nway_matching
N-way matching of segmented cell ROIs

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

# support
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support, as it is under active development. The community is welcome to submit issues, but you should not expect an active response.
