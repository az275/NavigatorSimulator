""" --------      Worker Machines Parameters      -------- """
GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 4
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

"""  --------       Workload Parameters    --------  """
TOTAL_NUM_OF_JOBS_PER_WORKFLOW = {0: 10000}

# TODO: STEP | LINEAR | EXPONENTIAL
SEND_RATES_BY_WORKFLOW = {"SEND_RATES": [95], "SEND_RATE_CHANGE_INTERVALS": [], "SEND_RATE_CHANGE_CURVES": []},

WORKLOAD_DISTRIBUTION = "POISON"  # UNIFORM | POISON | GAMMA

GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """
LOAD_INFORMATION_STALENESS = 1  # in ms

PLACEMENT_INFORMATION_STALENESS = 1  # in ms

RESCHEDULE_THREASHOLD = 1.5

"""  -------        Shepherd Parameters  --------- """
FLEX_LAMBDA = 3.03
HERD_K = 1.3
import numpy as np
HERD_PERIODICITY = np.inf
# TODO: decentralized scheduling w job abortion & realloc

"""  -------        General Scheduling Parameters  --------- """
ENABLE_MULTITHREADING = True # allow multiple models on same partition to run at once
ENABLE_MODEL_PREFETCH = False
ENABLE_DYNAMIC_MODEL_LOADING = True

# HERD | VORTEX | CUSTOM
# NOTE: HERD requires ENABLE_DYNAMIC_MODEL_LOADING
ALLOCATION_STRATEGY = 'HERD'

CUSTOM_ALLOCATION = [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# static experiment alloc, ppl1:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# static experiment alloc, ppl2:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]