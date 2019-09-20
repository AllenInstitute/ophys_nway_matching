# ophys_nway_matching
N-way matching of segmented cell ROIs

# quick start
(original code is python 2.7. TODO, update to python 3)

```
conda create -n testpy27 python=2.7
conda activate testpy27
pip install git+https://github.com/AllenInstitute/ophys_nway_matching
python -m nway.nway_matching --input_json tmp_example/input.json --output_json tmp_example/output.json
```

This creates `tmp_example/output.json`. The intent is that this file contains everything you need to know, both results and diagnostic metrics.

Some basic visualizations of the results are in `nway.diagnostics`
```
python -m nway.diagnostics --input_json tmp_example/output.json --output_pdf tmp_example/output.pdf

```

This should create `tmp_example/output.pdf`.
