""" --------      Worker Machines Parameters      -------- """
GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 8
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

"""  --------       Workload Parameters    --------  """
TOTAL_NUM_OF_JOBS_PER_WORKFLOW = {0: 5000, 1: 5000}

SEND_RATES_BY_WORKFLOW = {
    0: {"SEND_RATES": [55, 125],
        "SEND_RATE_CHANGE_INTERVALS": [1000],
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