""" --------      Worker Machines Parameters      -------- """
GPU_MEMORY_SIZE = 14000000  # in KB, 15BG for Tesla T4

TOTAL_NUM_OF_WORKERS = 140


"""  --------       Workload Parameters    --------  """
TOTAL_NUM_OF_JOBS = 10000

# The interval between two consecutive job creation events at each external client 
DEFAULT_CREATION_INTERVAL_PERCLIENT = 100     # ms. 

WORKLOAD_DISTRIBUTION = "POISON"  # UNIFORM | POISON | GAMMA

GAMMA_CV = 10  # Coefficient of variation for gamma distribution


"""  -------        Navigator Parameters  --------- """
LOAD_INFORMATION_STALENESS = 1  # in ms

PLACEMENT_INFORMATION_STALENESS = 1  # in ms

RESCHEDULE_THREASHOLD = 1.5

"""  -------        Scheduling Parameters  --------- """
BOOST_PARAMETER = 0.00104567474

# BOOST_PARAMETER = 0.0004137299