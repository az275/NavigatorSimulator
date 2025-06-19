""" --------      Worker Machines Parameters      -------- """
# GPU_MEMORY_SIZE = 24000000  # in KB, 24GB for NVIDIA A30

TOTAL_NUM_OF_NODES = 50
VALID_WORKER_SIZES = [24000000, 12000000, 6000000]

"""  --------       Workload Parameters    --------  """
TOTAL_NUM_OF_JOBS = 10000

# The interval between two consecutive job creation events at each external client 
DEFAULT_CREATION_INTERVAL_PERCLIENT = 18.2     # ms.

WORKLOAD_DISTRIBUTION = "POISON"  # UNIFORM | POISON | GAMMA

GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """
LOAD_INFORMATION_STALENESS = 1  # in ms

PLACEMENT_INFORMATION_STALENESS = 1  # in ms

RESCHEDULE_THREASHOLD = 1.5
