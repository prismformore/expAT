# Criterion: Softmax Loss + expAT Loss
python tools/train.py --config_file='configs/softmax_expAT.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/app/rgbir/rgbreid/data')" OUTPUT_DIR "('../log/market1501/expAT')"
