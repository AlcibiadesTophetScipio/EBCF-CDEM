train_dataset="astgtmv003"
# train_dataset="pyrenees"


declare -A model_dict
model_dict=(
  ################################### Astgtmv003
  ## RDN
  # [${train_dataset}_rdn-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-liif/runner.train.v1/epoch-605_psnr_pred-40.3059.pth"
  # [${train_dataset}_rdn-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-metasr/runner.train.v1/epoch-795_psnr_pred-40.2739.pth"
  # [${train_dataset}_rdn-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-sdfif-nearest/runner.train.v1/epoch-906_sr_loss-0.0115.pth"
  # [${train_dataset}_rdn-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-sdfif-bicubic/runner.train.v1/epoch-411_psnr-40.2548.pth"

  # ### RCAN
  # [${train_dataset}_rcan-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-liif/runner.train.v1/epoch-955_psnr_pred-40.2248.pth"
  # [${train_dataset}_rcan-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-metasr/runner.train.v1/epoch-979_psnr_pred-40.1283.pth"
  # [${train_dataset}_rcan-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-sdfif-nearest/runner.train.v1/epoch-710_psnr-40.4387.pth"
  # [${train_dataset}_rcan-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-sdfif-bicubic/runner.train.v1/epoch-631_psnr-40.7302.pth"

  # ### IMDN
  # [${train_dataset}_imdn-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-liif/runner.train.v1/epoch-536_psnr_pred-40.1434.pth"
  # [${train_dataset}_imdn-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-metasr/runner.train.v1/epoch-980_psnr_pred-40.0828.pth"
  # [${train_dataset}_imdn-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-sdfif-nearest/runner.train.v1/epoch-531_psnr-40.1967.pth"
  # [${train_dataset}_imdn-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-sdfif-bicubic/runner.train.v1/epoch-744_psnr-40.4236.pth"

  # ### SWINIR
  # [${train_dataset}_swinirC-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-liif/runner.train.v1/epoch-583_psnr_pred-40.0855.pth"
  # [${train_dataset}_swinirC-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-metasr/runner.train.v1/epoch-532_psnr_pred-40.1252.pth"
  # [${train_dataset}_swinirC-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-sdfif-nearest/runner.train.v1/epoch-939_psnr-40.1012.pth"
  # [${train_dataset}_swinirC-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-sdfif-bicubic/runner.train.v1/epoch-370_psnr-40.2807.pth"

  ################################### Pyrenees
  # ### RDN
  # [${train_dataset}_rdn-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-liif/runner.train.v1/epoch-727_psnr_pred-58.9440.pth"
  # [${train_dataset}_rdn-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-metasr/runner.train.v1/epoch-893_psnr_pred-59.1552.pth"
  # [${train_dataset}_rdn-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-sdfif-nearest/runner.train.v1/epoch-403_psnr-59.1072.pth"
  # [${train_dataset}_rdn-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rdn-sdfif-bicubic/runner.train.v1/epoch-846_psnr-59.2229.pth"

  # ### RCAN
  # [${train_dataset}_rcan-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-liif/runner.train.v1/epoch-871_psnr_pred-59.3105.pth"
  # [${train_dataset}_rcan-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-metasr/runner.train.v1/epoch-613_psnr_pred-59.0121.pth"
  # [${train_dataset}_rcan-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-sdfif-nearest/runner.train.v1/epoch-582_psnr-59.0578.pth"
  # [${train_dataset}_rcan-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_rcan-sdfif-bicubic/runner.train.v1/epoch-974_psnr-59.4398.pth"

  # ### IMDN
  # [${train_dataset}_imdn-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-liif/runner.train.v1/epoch-983_psnr_pred-58.6268.pth"
  # [${train_dataset}_imdn-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-metasr/runner.train.v1/epoch-575_psnr_pred-58.8184.pth"
  # [${train_dataset}_imdn-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-sdfif-nearest/runner.train.v1/epoch-839_psnr-58.8061.pth"
  # [${train_dataset}_imdn-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_imdn-sdfif-bicubic/runner.train.v1/epoch-311_psnr-58.7313.pth"

  # ### SWINIR
  # [${train_dataset}_swinirC-liif]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-liif/runner.train.v1/epoch-938_psnr_pred-58.7601.pth"
  # [${train_dataset}_swinirC-metasr]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-metasr/runner.train.v1/epoch-702_psnr_pred-58.4070.pth"
  # [${train_dataset}_swinirC-sdfif-nearest]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-sdfif-nearest/runner.train.v1/epoch-841_psnr-58.8089.pth"
  # [${train_dataset}_swinirC-sdfif-bicubic]="/data/syao/Exps/Solvent-encoders/${train_dataset}_swinirC-sdfif-bicubic/runner.train.v1/epoch-604_psnr-59.0484.pth"
  
)

for model_name in ${!model_dict[@]};
do
    echo "Model name: ${model_name}, model path: ${model_dict[${model_name}]}"
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