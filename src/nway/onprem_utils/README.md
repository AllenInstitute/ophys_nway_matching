### on-prem usage

* install as you normally would, but include the optional `ONPREM`:
```
pip install .[ONPREM]
```
or
```
pip install -e .[ONPREM]
```
The docker image will have the repo installed with `ONPREM`.

* you'll need the environment variables `LIMS_USER` and `LIMS_PASSWORD` defined as you would using AllenSDK or VisualBheaviorAnalysis.

* create an input json from a list of experiments:
```
$ python -m nway.onprem_utils.assembly \
    --experiment_ids 1023909329,1024382348,1025646202,1024072693 \
    --output_json tmp.json
INFO:OnPremInputAssembly:loaded 19 ROIs for experiment 1023909329
INFO:OnPremInputAssembly:loaded 20 ROIs for experiment 1024382348
INFO:OnPremInputAssembly:loaded 23 ROIs for experiment 1025646202
INFO:OnPremInputAssembly:loaded 21 ROIs for experiment 1024072693
INFO:OnPremInputAssembly:Wrote tmp.json
```

* use the just-created json to run nway cell matching:
```
$ python -m nway.nway_matching \
    --input_json tmp.json \
    --output_directory ./ \
    --output_json ./nway_out.json
```
