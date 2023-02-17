
Configure file system:
```
configs
├── config.yaml
├── datasets
│   ├── astgtmv003.yaml
│   ├── pyrenees.yaml
│   └── tyrol.yaml
├── experiments
│   ├── exp_debug.yaml
│   ├── exp_interpolation.yaml
│   ├── exp_liif-history.yaml
│   ├── exp_liif-sdf.yaml
│   ├── exp_liif.yaml
│   ├── exp_loss-all.yaml
│   ├── exp_loss-compos.yaml
│   ├── exp_loss-idpt.yaml
│   ├── exp_loss-new.yaml
│   ├── exp_mapif-enhance.yaml
│   ├── exp_mapif-sdf.yaml
│   ├── exp_mapif.yaml
│   ├── exp_metasr-sdf.yaml
│   ├── exp_metasr.yaml
│   ├── exp_sdfif.yaml
│   ├── exp_test128.yaml
│   └── exp_test.yaml
├── models
│   ├── edsr_baseline.yaml
│   ├── imdn.yaml
│   ├── interpolator.yaml
│   ├── liif.yaml
│   ├── mapif_sdf.yaml
│   ├── mapif.yaml
│   ├── metasr.yaml
│   ├── mlp.yaml
│   ├── rcan.yaml
│   ├── rdn.yaml
│   ├── sdfif.yaml
│   ├── swinirC.yaml
│   └── swinirL.yaml
└── optimizers
    ├── adam.yaml
    ├── multi_step_lr-1000.yaml
    └── multi_step_lr-300.yaml
```

Training cmd
```
python hydra_run.py exp_name=terrains_liif-sdf-x4 \
    run_spec.runner.type=runner.train.v1 \
    device="cuda:1" 

python hydra_run.py exp_name=astgtmv003_liif-sdf \
    datasets@dataset_spec=astgtmv003 \
    run_spec.runner.type=runner.train.v1 \
    device="cuda:0" 
    
```

Ground-truth tif generation.
```
python hydra_run.py --multirun +experiments=exp_interpolation model_spec.interpolation=bicubic,bilinear datasets@dataset_spec=astgtmv003,tyrol,pyrenees dataset_spec.test_dataset.scale_min=2,4,8,16

python hydra_run.py --multirun +experiments=exp_interpolation model_spec.interpolation=identity datasets@dataset_spec=astgtmv003,tyrol,pyrenees dataset_spec.test_dataset.scale_min=1

# change the code to get inp
python hydra_run.py --multirun +experiments=exp_interpolation model_spec.interpolation=bilinear datasets@dataset_spec=astgtmv003,tyrol,pyrenees dataset_spec.test_dataset.scale_min=2,4,8,16
```

Exps tif generation
```
bash multi-recs-run.sh
```

Generate error map in 3D
```
xvfb-run -a -s '-screen 0 1024x768x24' python plot_3d_metrics.py
```