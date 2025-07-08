""" --------      Worker Machines Parameters      -------- """
GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30
TOTAL_NUM_OF_NODES = 4
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

"""  --------       Workload Parameters    --------  """
TOTAL_NUM_OF_JOBS = 10000

SEND_RATES = [55, 125]
SEND_RATE_CHANGE_INTERVALS = [4000]
SEND_RATE_CHANGE_CURVES = ["STEP"] # TODO: STEP | LINEAR | EXPONENTIAL

WORKLOAD_DISTRIBUTION = "POISON"  # UNIFORM | POISON | GAMMA

GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """
LOAD_INFORMATION_STALENESS = 1  # in ms

PLACEMENT_INFORMATION_STALENESS = 1  # in ms

RESCHEDULE_THREASHOLD = 1.5

"""  -------        Shepherd Parameters  --------- """
FLEX_LAMBDA = 3.03
HERD_K = 1.3