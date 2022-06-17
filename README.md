# GenLabel

This repo contains demo PyTorch reimplementations of GenLabel on both synthetic & real datasets.

## Dependencies

Tested stable dependencies:

* Ubuntu 18.04.5
* python 3.7.10 (Anaconda)
* PyTorch 1.10.0+cpu
* numpy 1.20.3
* scikit-learn 0.24.2
* scipy 1.6.2
* matplotlib 3.4.2
* openml 0.12.2
* pandas 1.3.3
* jupyter 1.0.0


An alternative way to set up the environment is running the following command with our `requirements.txt`.

```
pip install -r requirements.txt
```

## OpenML

Please follow the instruction in OpenML/ directory

## 2D cube

### Clean validation

#### Vanilla training
```
python 2dcube_model.py --mixup_setting 0 --lr 0.1 --epoch 40 --validate cln --num_sample 20
```

#### Mixup
```
python 2dcube_model.py --mixup_setting 1 --lr 0.1 --epoch 40 --validate cln --num_sample 20
```

#### Mixup + GenLabel
```
python 2dcube_model.py --mixup_setting 2 --lr 0.1 --epoch 40 --lam 1 --validate cln --num_sample 20
```

## 3D cube

### Clean validation

#### Vanilla training
```
python 3dcube_model.py --mixup_setting 0 --lr 0.1 --epoch 40 --validate cln --num_sample 20
```

#### Mixup
```
python 3dcube_model.py --mixup_setting 1 --lr 0.1 --epoch 40 --validate cln --num_sample 20
```

#### Mixup + GenLabel
```
python 3dcube_model.py --mixup_setting 2 --lr 0.1 --epoch 40 --lam 0.8 --validate cln --num_sample 20
```

## 9-class Gaussian

For 9-class Gaussian dataset, please find details in [this notebook](9-class_Gaussian.ipynb).

## Moon

### Clean validation

#### Vanilla training
```
python Syn_model.py --mixup_setting 0 --dataset moon --lr 0.1 --epoch 100 --validate cln --num_sample 1000
```

#### Mixup
```
python Syn_model.py --mixup_setting 1 --dataset moon --lr 0.1 --epoch 100 --validate cln --num_sample 1000
```

#### Mixup + GenLabel
```
python Syn_model.py --mixup_setting 2 --dataset moon --lr 0.1 --epoch 100 --lam 1 --bw 0.2 --validate cln --num_sample 1000
```

## Circle

### Clean validation

#### Vanilla training
```
python Syn_model.py --mixup_setting 0 --dataset circle --lr 0.1 --epoch 100 --validate cln --num_sample 1000
```

#### Mixup
```
python Syn_model.py --mixup_setting 1 --dataset circle --lr 0.1 --epoch 100 --validate cln --num_sample 1000
```

#### Mixup + GenLabel
```
python Syn_model.py --mixup_setting 2 --dataset circle --lr 0.1 --epoch 100 --lam 0.8 --bw 0.2 --validate cln --num_sample 1000
```

## Two-circle

### Clean validation

#### Vanilla training
```
python Syn_model.py --mixup_setting 0 --dataset twocircle --lr 0.1 --epoch 100 --validate cln --num_sample 1000
```

#### Mixup
```
python Syn_model.py --mixup_setting 1 --dataset twocircle --lr 0.1 --epoch 100 --validate cln --num_sample 1000
```

#### Mixup + GenLabel
```
python Syn_model.py --mixup_setting 2 --dataset twocircle --lr 0.1 --epoch 100 --lam 1 --bw 0.1 --validate cln --num_sample 1000
```