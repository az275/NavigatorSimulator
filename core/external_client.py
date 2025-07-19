from core.job import *
from core.config import *
import numpy as np
from core.config import *


class ExternalClient(object):

    def __init__(self, simulation, job_type=0):
        self.simulation = simulation
        self.job_type = job_type

    def create_job(self, current_time, current_job_id, send_rate, slo):
        job_create_delay = 1 / send_rate * 1000
        if WORKLOAD_DISTRIBUTION == "POISON":
            job_create_delay = np.random.poisson(lam=(1 / send_rate * 1000))
        if WORKLOAD_DISTRIBUTION == "GAMMA":
            shape = (1 / send_rate * 1000)**2 / GAMMA_CV**2
            scale = GAMMA_CV**2 / (1 / send_rate * 1000)
            job_create_delay = np.random.gamma(shape, scale)
        job = Job(create_time=current_time + job_create_delay,
                  job_type_id=self.job_type, job_id=current_job_id, slo=slo)
        job = self.log_job_creation_time(job, current_time + job_create_delay)
        return job

    def select_initial_worker_id(self):
        initial_worker_id = None
        initial_worker_id = np.random.choice(
            range(self.simulation.total_workers))
        return initial_worker_id

    def log_job_creation_time(self, job, creation_time):
        for task in job.tasks:
            task.log.job_creation_timestamp = creation_time
        return job
