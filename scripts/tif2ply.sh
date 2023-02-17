
function scan_dir_and_trans_tif2dae() {
    # for tif_file in $(ls $1 | head -1); # debug
    for tif_file in $(ls $1);
    do
        tif2dae_cmd="python DTM2MESH.py -input $1${tif_file} 
                    -output $2${tif_file%.tif}.dae
                    -resolution $3
                    "
        echo $tif2dae_cmd
        eval $tif2dae_cmd

        dae2ply_cmd="xvfb-run -a 
                    -s '-screen 0 800x600x24'
                    meshlabserver
                    -i $2${tif_file%.tif}.dae
                    -o $4${tif_file%.tif}.ply
                    "
        echo $dae2ply_cmd
        eval $dae2ply_cmd
    done
}

main_dir='/data/syao/Exps/Solvent-tif'
save_dae_dir='/data/syao/Exps/Solvent-dae'
save_ply_dir='/data/syao/Exps/Solvent-ply'

scale=2
# scale=4
# scale=8
# scale=16

# train_dataset="astgtmv003"
train_dataset="pyrenees"

# test_dataset='astgtmv003'
# test_dataset='pyrenees'
test_dataset='tyrol'

# solo recs
scan_subdirs=(
    # gt
    # "${main_dir}/multi-exp_interpolator-gt/${test_dataset}-identity-x1/*/*/"
    # interpolation
    # "${main_dir}/multi-exp_interpolator/${test_dataset}-bilinear-x${scale}/*/*/"
    # "${main_dir}/multi-exp_interpolator/${test_dataset}-bicubic-x${scale}/*/*/"

    # # mapif
    # "${main_dir}/multi-${test_dataset}_mapif-global/${test_dataset}-x${scale}/*/*/"
    # "${main_dir}/multi-${test_dataset}_mapif-local/${test_dataset}-x${scale}/*/*/"
    # "${main_dir}/multi-${test_dataset}_mapif-sdf_loss-compos/${test_dataset}-x${scale}/*/*/"
    # # metasr
    # "${main_dir}/dem-pro-recs/${test_dataset}-edsr-metasr_base-x${scale}/"
    # "${main_dir}/multi-${test_dataset}_metasr-sdf_loss-compos/${test_dataset}-x${scale}/*/*/"
    # # liif
    # "${main_dir}/dem-pro-recs/${test_dataset}-edsr-liif_naive-x${scale}/"
    # "${main_dir}/dem-pro-recs/${test_dataset}-edsr-liif_base-x${scale}/"
    # "${main_dir}/multi-${test_dataset}_liif-sdf_loss-compos/${test_dataset}-x${scale}/*/*/"
    # # sdfif
    # "${main_dir}/multi-${test_dataset}_sdfif-nearest/${test_dataset}-x${scale}/*/*/"
    # "${main_dir}/multi-${test_dataset}_sdfif-bicubic/${test_dataset}-x${scale}/*/*/"
)

# cross recs
# scan_subdirs=(
#     # # mapif
#     "${main_dir}/multi-${train_dataset}_mapif-global/${test_dataset}-x${scale}/*/*/"
#     "${main_dir}/multi-${train_dataset}_mapif-local/${test_dataset}-x${scale}/*/*/"
#     "${main_dir}/multi-${train_dataset}_mapif-sdf_loss-compos/${test_dataset}-x${scale}/*/*/"
#     # # metasr
#     "${main_dir}/dem-pro-recs/${train_dataset}To${test_dataset}-edsr-metasr_base-x${scale}/"
#     "${main_dir}/multi-${train_dataset}_metasr-sdf_loss-compos/${test_dataset}-x${scale}/*/*/"
#     # # liif
#     # "${main_dir}/dem-pro-recs/${train_dataset}To${test_dataset}-edsr-liif_naive-x${scale}/"
#     "${main_dir}/dem-pro-recs/${train_dataset}To${test_dataset}-edsr-liif_base-x${scale}/"
#     "${main_dir}/multi-${train_dataset}_liif-sdf_loss-compos/${test_dataset}-x${scale}/*/*/"
#     # # sdfif
#     "${main_dir}/multi-${train_dataset}_sdfif-nearest/${test_dataset}-x${scale}/*/*/"
#     "${main_dir}/multi-${train_dataset}_sdfif-bicubic/${test_dataset}-x${scale}/*/*/"
# )

for itr_d in "${scan_subdirs[@]}"; do
    # echo "$(${itr_d})"
    # echo "$(ls ${itr_d})"
    parse_d=$(echo ${itr_d})
    if [ -d $parse_d ]; then
    echo "ls ${itr_d}"

    save_dae_subdir=${itr_d/${main_dir}/${save_dae_dir}}
    save_dae_subdir=${save_dae_subdir%"*/*/"}
    mkdir -p ${save_dae_subdir}
    echo ${save_dae_subdir}

    save_ply_subdir=${itr_d/${main_dir}/${save_ply_dir}}
    save_ply_subdir=${save_ply_subdir%"*/*/"}
    mkdir -p ${save_ply_subdir}
    echo ${save_ply_subdir}
    
    scan_dir_and_trans_tif2dae ${itr_d} ${save_dae_subdir} 1 ${save_ply_subdir}
    fi
done