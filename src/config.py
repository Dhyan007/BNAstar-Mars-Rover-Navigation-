import torch

# Original project paths — intentionally unchanged.
IMAGES_PATH = '/Users/dhyan/Desktop/project/BNA*-mars-rover/ai4mars-dataset-merged-0.1/msl/images/edr'
MASK_PATH_TEST = '/Users/dhyan/Desktop/project/BNA*-mars-rover/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min1-100agree'
MODEL_PATH = '/Users/dhyan/Desktop/project/BNA*-mars-rover/ai4mars_fcn_resnet50_state_dict.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# A* SEARCH SPACE SETTINGS
# ==========================================
LAYER_SIZES = [1, 2, 4, 8, 16, 8, 4, 2, 1]
NUM_NODES = sum(LAYER_SIZES)

START_NODE = 0
GOAL_NODE = NUM_NODES - 1

BRANCHING_FACTOR = 2
