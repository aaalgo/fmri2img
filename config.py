N_TRAIN = 9000      # training images, total image is 10000 or slightly less

SUBJECT = 1
ANAT_SPACE = 'anat0pt8'
FUNC_SPACE = 'func1pt8'

VISUAL_ROI_INPUT = 'prf-visualrois.nii.gz'

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
ADAM_WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
DATALOADER_NUM_WORKERS=16
LR_SCHEDULER = 'constant'

TRAIN_TIMESTEPS = 50

VISUAL_DIM = 4660
DIM = 4660
ENCODE_DIM = 768

try:
    with open('local_config.py', 'r') as f:
        code = f.read()
    exec(code)
except:
    pass

VISUAL_ROI_PATH = 'data/visual%02d.nii.gz' % SUBJECT

