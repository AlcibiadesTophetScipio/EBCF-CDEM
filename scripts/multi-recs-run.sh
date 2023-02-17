
declare -A model_dict
model_dict=(
  # # Train on Pyreness Dataset
  # [pyrenees_mapif-global]="/data/syao/Exps/Solvent/Allies/pyrenees_mapif-global/epoch-996_psnr-58.1650.pth"
  # [pyrenees_mapif-local]="/data/syao/Exps/Solvent/Allies/pyrenees_mapif-local/epoch-764_psnr-59.1910.pth"
  # [pyrenees_mapif-sdf_loss-compos]="/data/syao/Exps/Solvent/Allies/pyrenees_mapif-sdf_loss-compos/epoch-893_psnr_enhance-58.8380.pth"
  # [pyrenees_metasr-sdf_loss-compos]="/data/syao/Exps/Solvent/Allies/pyrenees_metasr-sdf_loss-compos/epoch-613_psnr_enhance-58.9805.pth"
  # [pyrenees_liif-sdf_loss-compos]="/data/syao/Exps/Solvent/Allies/pyrenees_liif-sdf_loss-compos/epoch-977_psnr_enhance-59.1153.pth"
  # [pyrenees_sdfif-nearest]="/data/syao/Exps/Solvent/Allies/pyrenees_sdfif-nearest/epoch-929_psnr-59.0466.pth"
  # [pyrenees_sdfif-bicubic]="/data/syao/Exps/Solvent/Allies/pyrenees_sdfif-bicubic/epoch-847_psnr-59.3613.pth"

  # # Train on Astgtmv003 Dataset
  # [astgtmv003_mapif-global]="/data/syao/Exps/Solvent/Allies/astgtmv003_mapif-global/epoch-575_psnr-40.2821.pth"
  # [astgtmv003_mapif-local]="/data/syao/Exps/Solvent/Allies/astgtmv003_mapif-local/epoch-736_psnr-39.9833.pth"
  # [astgtmv003_mapif-sdf_loss-compos]="/data/syao/Exps/Solvent/Allies/astgtmv003_mapif-sdf_loss-compos/epoch-915_psnr_enhance-40.1155.pth"
  # [astgtmv003_metasr-sdf_loss-compos]="/data/syao/Exps/Solvent/Allies/astgtmv003_metasr-sdf_loss-compos/epoch-812_psnr_enhance-39.9991.pth"
  # [astgtmv003_liif-sdf_loss-compos]="/data/syao/Exps/Solvent/Allies/astgtmv003_liif-sdf_loss-compos/epoch-710_loss-0.0236.pth"
  # [astgtmv003_sdfif-nearest]="/data/syao/Exps/Solvent/Allies/astgtmv003_sdfif-nearest/epoch-818_psnr-40.1705.pth"
  # [astgtmv003_sdfif-bicubic]="/data/syao/Exps/Solvent/Allies/astgtmv003_sdfif-bicubic/epoch-485_psnr-40.4172.pth"

  # Remake
  # [re-pyrenees_metasr]="/data/syao/Exps/Solvent/pyrenees_metasr/runner.train.v1/epoch-463_psnr_pred-59.1795.pth"
  # [re-pyrenees_metasr-sdf]="/data/syao/Exps/Solvent/pyrenees_metasr-sdf/runner.train.v1/epoch-887_psnr_enhance-59.1016.pth"
  # [re-pyrenees_liif]="/data/syao/Exps/Solvent/pyrenees_liif/runner.train.v1/epoch-991_psnr_pred-58.9555.pth"
  # [re-pyrenees_liif-sdf]="/data/syao/Exps/Solvent/pyrenees_liif-sdf/runner.train.v1/epoch-452_psnr_enhance-58.9738.pth"
  [re-pyrenees_sdfif-bicubic]="/data/syao/Exps/Solvent/pyrenees_sdfif-bicubic/runner.train.v1/epoch-757_psnr-59.1465.pth"
  
  # [re-astgtmv003_liif]="/data/syao/Exps/Solvent/astgtmv003_liif/runner.train.v1/epoch-821_psnr_pred-40.2290.pth"
  # [re-astgtmv003_metasr]="/data/syao/Exps/Solvent/astgtmv003_metasr/runner.train.v1/epoch-567_psnr_pred-40.1243.pth"
  [re-astgtmv003_sdfif-bicubic]="/data/syao/Exps/Solvent/astgtmv003_sdfif-bicubic/runner.train.v1/epoch-737_psnr-40.4464.pth"

)

for model_name in ${!model_dict[@]};
do
    # echo "Model name: ${model_name}, model path: ${model_dict[${model_name}]}"
    task_cmd="python hydra_run.py --multirun 
    +experiments=exp_test128 
    datasets@dataset_spec=tyrol,pyrenees,astgtmv003 
    dataset_spec.test_dataset.scale_min=2,4,8,16 
    exp_name=${model_name} 
    test_spec.model_pth=${model_dict[${model_name}]}
    "

    echo $task_cmd
    $task_cmd
done