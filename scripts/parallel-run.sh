
dataset=("astgtmv003" "pyrenees")
scales=(2 4 8 16)
for train_dataset in "${dataset[@]}"; do
    for test_dataset in "${dataset[@]}"; do
        for scale in "${scales[@]}"; do
            cmd_string="bash script-test.sh ${train_dataset} ${test_dataset} ${scale} &"
            eval "$cmd_string" 
            if [ "$?" = "1" ]; then
                echo "Dont run ${cmd_string}"
            fi
        done
        
    done
done

wait

