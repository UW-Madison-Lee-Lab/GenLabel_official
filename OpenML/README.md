## Train/test on OpenML

### Run various schemes 

```
python main.py \
    -a, --arch [log_reg | 3FC, whether use logistic regression of 3-layer FC nets, default: log_reg] \
    -m, --mode [vanilla | mixup | adamixup | gmlabel | kdelabel | bestlabel, method to run, can choose multiple in one run] \
    -p, --partition [a specific list of datasets to run, the partition id and the list of datasets can be found in function get_did_list in utils.py] \
    --did [dataset id to run, if not used then run for all datasets] \
    --meansure [cln | rob, whether run for clean accuracy or robust accuracy, default: cln] \
```

Results will be listed in csv/ folder.
