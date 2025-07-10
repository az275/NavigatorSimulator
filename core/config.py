""" --------      Worker Machines Parameters      -------- """
GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 4
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

"""  --------       Workload Parameters    --------  """
TOTAL_NUM_OF_JOBS_PER_WORKFLOW = {0: 10000, 1: 0}

SEND_RATES_BY_WORKFLOW = {
    0: {"SEND_RATES": [55, 125, 95],
        "SEND_RATE_CHANGE_INTERVALS": [1000, 3000],
        "SEND_RATE_CHANGE_CURVES": []}, # TODO: STEP | LINEAR | EXPONENTIAL
    1: {"SEND_RATES": [24, 6, 16],
        "SEND_RATE_CHANGE_INTERVALS": [2000, 2000],
        "SEND_RATE_CHANGE_CURVES": []},
}

WORKLOAD_DISTRIBUTION = "POISON"  # UNIFORM | POISON | GAMMA

GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """
LOAD_INFORMATION_STALENESS = 1  # in ms

PLACEMENT_INFORMATION_STALENESS = 1  # in ms

RESCHEDULE_THREASHOLD = 1.5

"""  -------        Shepherd Parameters  --------- """
FLEX_LAMBDA = 3.03
HERD_K = 1.3
HERD_PERIODICITY = 10000   # runs HERD every [HERD_PERIODICITY] ms

"""  -------        General Scheduling Parameters  --------- """
ENABLE_DYNAMIC_MODEL_LOADING = True

# HERD | VORTEX | CUSTOM
# NOTE: HERD requires ENABLE_DYNAMIC_MODEL_LOADING
ALLOCATION_STRATEGY = "HERD"

"""
    If ALLOCATION_STRATEGY == "CUSTOM", must define CUSTOM_ALLOCATION : list[tuple[int, list[int]]]
    where each (int, list[int]) is the (partition_size, list[model ids that can be loaded to partition])
    of a worker.

    Currently: If dynamic, may load from outside list

    TODO:

    If ENABLE_DYNAMIC_LOADING, then models will be greedily preloaded onto each worker until space
    runs out, and when a task is scheduled, the worker may evict/load new models from the list.

    Otherwise, workers will attempt to preload all models in the list for a static allocation.
"""
CUSTOM_ALLOCATION = [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2]),
                     (12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]

# static experiment alloc, ppl1:
# [(24, [1]), (24, [1]), (24, [1]), (6, [3]), (6, [3]), (6, [3]), (6, [0, 2])]

# static experiment alloc, ppl2:
# [(12, [4]), (12, [5,6]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7]), (12, [7])]