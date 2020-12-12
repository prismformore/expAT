# Criterion: Softmax Loss + expAT Loss
python tools/train.py --config_file='configs/softmax_expAT.yml' MODEL.DEVICE_ID "('2')" DATASETS.NAMES "('dukemtmc')" DATASETS.ROOT_DIR "('/app/rgbir/rgbreid/data')" OUTPUT_DIR "('../log/duke/expAT')"

