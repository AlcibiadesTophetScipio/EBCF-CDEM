# EBCF-CDEM

## Description
A continuous DEM model.

## Installation
```
pip install -r requirements.txt
```

## Instructions

Our code deeply relys on the "hydra" package, for every experiments it need to modify the corresponding config file. We provid the debug template for config files.
```
python hydra_run.py --cfg job --resolve ${config_file}

```

### I. Prepare Data
1. Download the data:
[TFASR30 and TFASR30to10 datasets](https://doi.org/10.6084/m9.figshare.19225374)
[Pyrenees and Tyrol datasets](https://www.virvig.eu/fcn-terrains/terrains.zip)

2. Split Pyrenees and Tyrol datasets:
```
python make_dataset.py --dirRawDataset ${data_path}/terrain/pyrenees
# It should generate the split files in "./temp" dir. Move and rename the dir if you want. Similar processing for Tyrol dataset.
```

3. Generate json file for datasets: Modify your specific changes in "generate_json_for_dataset.py" file.

4. Modify the config file in "configs/datasets/${dataset_name}.yaml"

5. Generate the training and testing data. Note that you should specify the "run_dir" in "configs/exp_interpolation.yaml"
```
python hydra_run.py --multirun \
        experiments=exp_interpolation \
        datasets@dataset_spec=tyrol,pyrenees,tfasr \
        model_spec.interpolation=identity \
        dataset_spec.test_dataset.scale_min=1
```

### II. Test

#### Using our trained model
Using the checkpoint of TFASR30 to generate super-resolution DEMs. Note that you should specify the "run_dir" in "configs/exp_test.yaml".
```
python hydra_run.py --multirun \
     experiments=exp_test \
     datasets@dataset_spec=tfasr \
     dataset_spec.test_dataset.scale_min=2,4 \
     exp_name="tfasr_edsrB-ebcf-nearest-pe16_best" \
     test_spec.model_pth="./checkpoints/TFASR30_ebcf-nearest-pe16_best-epoch.pth" \
     device='cuda:0'
```
Similar to Pyrenees:
```
python hydra_run.py --multirun \
     experiments=exp_test \
     datasets@dataset_spec=pyrenees \
     dataset_spec.test_dataset.scale_min=2,4,6,8 \
     exp_name="pyrenees_edsrB-ebcf-nearest-pe16_best" \
     test_spec.model_pth="./checkpoints/Pyrenees_ebcf-nearest-pe16_best-epoch.pth" \
     device='cuda:0'
```
and TFASR30to10:
```
python hydra_run.py --multirun \
     experiments=exp_test \
     datasets@dataset_spec=tfasr_30to10 \
     dataset_spec.test_dataset.scale_min=3 \
     exp_name="tfasr30to10_edsrB-ebcf-nearest-pe16_best" \
     test_spec.model_pth="./checkpoints/TFASR30to10_ebcf-nearest-pe16_best-epoch.pth" \
     device='cuda:0'
```

#### Calculate metrics
For calculating metrics, you shoud specify the "gt_dir" and the "sr_dir" in "sr-tif.yaml". Note that the "gt_dir" means the "run_dir" in "configs/exp_interpolation.yaml" but shoud be more specific for the dataset. The "sr_dir" means the generated results of the super-resolution model.
Also, it can use a flexible way to define *vars* in "sr-tif.yaml". More details please refer to the useage of "OmegaConf".
Now, just run:
```
python cal_dem_metrics.py
```



### III. Train your model

#### On TFASR30 dataset
Without the bias prediction:
```
python hydra_run.py \
        experiments=exp_ebcf \
        device='cuda:0' \
        datasets@dataset_spec=tfasr \
        model_spec.interp_mode='none' \
        dataset_spec.train_dataset.dataset.repeat=4 
```

Without the pos encoding:
```
python hydra_run.py \
        experiments=exp_ebcf \
        device='cuda:0' \
        datasets@dataset_spec=tfasr \
        model_spec.interp_mode='nearest' \
        dataset_spec.train_dataset.dataset.repeat=4
```

With the pos encoding:
```
python hydra_run.py \
        experiments=exp_ebcf-pe \
        device='cuda:0' \
        datasets@dataset_spec=tfasr \
        model_spec.interp_mode='nearest' \
        dataset_spec.train_dataset.dataset.repeat=4 \
        model_spec.posEmbeder.spec.n_harmonic_functions=16
```

#### On TFASR30to10 dataset

```
python hydra_run.py \
        experiments=exp_ebcf-pe \
        device='cuda:1' \
        datasets@dataset_spec=tfasr_30to10 \
        model_spec.interp_mode='nearest' \
        model_spec.posEmbeder.spec.n_harmonic_functions=16
```