SUBJECT = 1
ANAT_SPACE = 'anat0pt8'
FUNC_SPACE = 'func1pt8'

VISUAL_ROI_INPUT = 'prf-visualrois.nii.gz'
VISUAL_ROI_PATH = 'data/visual.nii.gz'

NSD_ROOT = 'data/raw'

SIGMA_THRESHOLD = 60
EDGE_THRESHOLD = 0.5

#IMAGE_SIZE = 128

MIXED_PRECISION = 'fp16'
REPORT_TO = 'wandb'
SCALE_LR = False
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = 64
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 0.1
ADAM_EPSILON = 1e-8
DATALOADER_NUM_WORKERS=16
LR_SCHEDULER = 'constant'

NEIGHBORS_DELTA = []

TRAIN_TIMESTEPS = 50

RESPONSE_DELAY = 1

VISUAL_DIM = 4660
DIM = 4660
ENCODE_DIM = 768

ROI_PATH = 'data/subj01_visual.nii.gz'

for d0 in [-1, 0, 1]:
    for d1 in [-1, 0, 1]:
        for d2 in [-1, 0, 1]:
            if d0 == 0 and d1 == 0 and d2 == 0:
                pass
            NEIGHBORS_DELTA.append((d0, d1, d2))

try:
    with open('local_config.py', 'r') as f:
        code = f.read()
    exec(code)
except:
    pass

