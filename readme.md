# MiSH

MiSH (Minor-Subspace Householder Reflection Adaptation).

## Dataset preparation

Follow the dataset access instructions linked in [IRRA's dataset preparation guide](https://github.com/anosorae/IRRA#prepare-datasets), then organize the files as expected by this project's loaders:

```text
Datasets/
├── CUHK-PEDES/
│   ├── imgs/
│   └── reid_raw.json
├── ICFG-PEDES/
│   ├── imgs/
│   └── ICFG-PEDES.json
└── RSTPReid/
    ├── imgs/
    └── data_captions.json
```

Image paths inside the annotations must resolve relative to the corresponding `imgs/` directory. The ICFG annotation filename expected here is **`ICFG-PEDES.json`**.

## Training

Run commands from the project root. The following commands use the fixed defaults and explicitly select the dataset and its threshold:

```bash
python train.py --root_dir /path/to/Datasets --dataset_name RSTPReid --hra_subspace 0.9
python train.py --root_dir /path/to/Datasets --dataset_name CUHK-PEDES --hra_subspace 0.5
python train.py --root_dir /path/to/Datasets --dataset_name ICFG-PEDES --hra_subspace 0.2
```

## Evaluation

Set the `sub` variable in `test.py` to the directory containing the desired run's `configs.yaml`, `best.pth`, and/or `last.pth`, then run:

```bash
python test.py
```