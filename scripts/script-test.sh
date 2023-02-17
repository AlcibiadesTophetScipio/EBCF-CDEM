

if [ $# -eq 3 ]; then
    train_dataset=$1
    test_dataset=$2
    scale=$3
    if [ ${train_dataset} = ${test_dataset} ]; then
        train_dataset=""
        exit 1
    fi
elif [ $# -eq 2 ]; then
    test_dataset=$1
    scale=$2
fi

if [ -z ${train_dataset} ]; then
    echo ${test_dataset}_${scale}
else
    echo ${train_dataset}_${test_dataset}_${scale}
fi

sleep 3

