
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

main_dir='/data/syao/Exps/Solvent-encoders-tif'
save_dae_dir='/data/syao/Exps/Solvent-encoders-dae'
save_ply_dir='/data/syao/Exps/Solvent-encoders-ply'

train_dataset="astgtmv003"
# train_dataset="pyrenees"
test_dataset=${train_dataset}

# scale=2
# scale=4
# scale=8
scale=16


# encoder="rdn"
# encoder="rcan"
# encoder="imdn"
encoder="swinirC"

scan_subdirs=(
    "${main_dir}/multi-${train_dataset}_${encoder}-metasr/${test_dataset}-x${scale}/*/*/"
    "${main_dir}/multi-${train_dataset}_${encoder}-liif/${test_dataset}-x${scale}/*/*/"
    "${main_dir}/multi-${train_dataset}_${encoder}-sdfif-nearest/${test_dataset}-x${scale}/*/*/"
    "${main_dir}/multi-${train_dataset}_${encoder}-sdfif-bicubic/${test_dataset}-x${scale}/*/*/"
)

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