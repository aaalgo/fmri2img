TIMESERIES_SHAPE = None # (81, 104, 83, 226)
CORR_INPUT_PATTERN = None

SIGMA_THRESHOLD = 60
EDGE_THRESHOLD = 0.5


NEIGHBORS_DELTA = []

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

