# Criterion: Softmax Loss + Triplet Loss
python tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('4')" DATASETS.NAMES "('dukemtmc')" DATASETS.ROOT_DIR "('/app/rgbir/rgbreid/data')" OUTPUT_DIR "('../log/duke/triplet')"