# Criterion: Softmax Loss + Triplet Loss (BNNECK)
python tools/train.py --config_file='configs/softmax_bnneck.yml' MODEL.DEVICE_ID "('3')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/app/rgbir/rgbreid/data')" OUTPUT_DIR "('../log/market1501/bnneck')"
