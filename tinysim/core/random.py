import numpy as np


RANDOM_GEN = np.random.default_rng()
SEED = None


def set_seed(seed):
  global RANDOM_GEN
  SEED = seed
  RANDOM_GEN = np.random.default_rng(seed)

def get_seed():
  return SEED