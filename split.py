#!/usr/bin/env python3
import pickle
import random

with open('data/samples.pkl', 'rb') as f:
    samples = pickle.load(f)

random.shuffle(samples)

N_TRAIN = 3000

train = samples[:N_TRAIN]
test = samples[N_TRAIN:]

with open('data/train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('data/test.pkl', 'wb') as f:
    pickle.dump(test, f)


