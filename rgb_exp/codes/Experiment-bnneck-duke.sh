# Criterion: Softmax Loss + Triplet Loss (BNNECK)
python tools/train.py --config_file='configs/softmax_bnneck.yml' MODEL.DEVICE_ID "('6')" DATASETS.NAMES "('dukemtmc')" DATASETS.ROOT_DIR "('/app/rgbir/rgbreid/data')" OUTPUT_DIR "('../log/duke/bnneck')" 

