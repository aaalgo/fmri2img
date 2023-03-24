TIMESERIES_SHAPE = None # (81, 104, 83, 226)
CORR_INPUT_PATTERN = None

SIGMA_THRESHOLD = 60
EDGE_THRESHOLD = 0.5

IMAGE_SIZE = 512

MIXED_PRECISION = 'fp16'
REPORT_TO = 'wandb'
SCALE_LR = False
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = 12
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
DATALOADER_NUM_WORKERS=8
LR_SCHEDULER = 'constant'

NEIGHBORS_DELTA = []

TRAIN_TIMESTEPS = 50

RESPONSE_DELAY = 1
VISUAL_DIM = 3270
DIM = VISUAL_DIM
ENCODE_DIM = 768

EARLY_VISUAL_CORTEX = [1021, 1001, 1002, 2002, 2003, 2001]
APARC_PATH = 'data/aparc_func1pt8.nii.gz'

for d0 in [-1, 0, 1]:
    for d1 in [-1, 0, 1]:
        for d2 in [-1, 0, 1]:
            if d0 == 0 and d1 == 0 and d2 == 0:
                pass
            NEIGHBORS_DELTA.append((d0, d1, d2))

try:
    from local_config import *
    print("Found local config, importing...")
except:
    pass

