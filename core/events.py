import random
from core.job import *
from core.network import *
import numpy as np
from core.config import *


class Event(object):
    """ Abstract class representing events. """

    def __init__(self):
        raise NotImplementedError("Event is an abstract class and cannot be "
                                  "instantiated directly")

    def run(self, current_time):
        """ Returns any events that should be added to the queue. """
        raise NotImplementedError("The run() method must be implemented by "
                                  "each class subclassing Event")

    def to_string(self, current_time):
        """ Returns the string describing the event """
        raise NotImplementedError("The to_string() method must be implemented by "
                                  "each class subclassing Event")


class JobCreationAtExternalClient(Event):
    """
    Event signifying that a Job is created at External client
    JobCreationAtExternalClient(s) Events 
    1. generate one another in a chain reaction.
    2. followup a JobArrival___ event for cloud to execute.
    """
    job_creation_counter = 0  # static variable

    def __init__(self, simulation, external_client_id):
        self.simulation = simulation
        self.external_client_id = external_client_id
        self.job_id = JobCreationAtExternalClient.job_creation_counter
        JobCreationAtExternalClient.job_creation_counter += 1
        
    def run(self, current_time):
        job, creation_delay = self.simulation.external_clients[self.external_client_id].create_job(
             current_time, self.job_id)
        self.simulation.jobs[job.id] = job  # tracking purpos
        new_events = []
        if self.job_creation_counter > 10 + TOTAL_NUM_OF_JOBS:
            return new_events
        # 1.  JobCreationAtExternalClient(s) generate one another in a chain reaction
        new_events.append(EventOrders(current_time + creation_delay,
                          JobCreationAtExternalClient(self.simulation, self.external_client_id)))
        # 2. create a JobArrivalAtScheduler event
        if(self.simulation.centralized_scheduler):
            new_events.append(EventOrders(current_time + CPU_to_CPU_delay(job.tasks[0].input_size),
                                          JobArrivalAtScheduler(self.simulation, job)))
            # new_events.append(EventOrders(current_time + UplinkEdgeToCloud_delay(job.tasks[0].input_size),
            #                               JobArrivalAtScheduler(self.simulation, job)))
        else:
            initial_worker_id = self.simulation.external_clients[self.external_client_id].select_initial_worker_id()
            new_events.append(EventOrders(current_time + CPU_to_CPU_delay(job.tasks[0].input_size),
                                          JobArrivalAtWorker(self.simulation, job, initial_worker_id)))
            # new_events.append(EventOrders(current_time + UplinkEdgeToCloud_delay(job.tasks[0].input_size),
            #                               JobArrivalAtWorker(self.simulation, job, initial_worker_id)))
        return new_events

    def to_string(self):
        return "[Job Creation at Client (Job {})] ++".format(self.job_id)


class JobArrivalAtScheduler(Event):
    """
    Event signifying that a Job arrived to a Centralized scheduler.
    Only for Centralized Schedulers
    """

    def __init__(self, simulation, job):
        self.simulation = simulation
        self.job = job

    def run(self, current_time):
        # Schedule job
        if self.simulation.job_split == "PER_TASK":
            new_events = self.simulation.schedule_job_and_send_tasks(
                self.job, current_time)
        elif self.simulation.job_split == "PER_JOB":
            new_events = self.simulation.schedule_job_and_send_job(
                self.job, current_time)
        return new_events

    def to_string(self):
        return "[Job Arrival at Scheduler (Job {})] ++".format(self.job.id)


class JobArrivalAtWorker(Event):
    """
    Event signifying that a Job arrived to a Cascade node (a Worker).
    Only for Decentralized Schedulers
    """

    def __init__(self, simulation, job, worker_id):
        self.simulation = simulation
        self.worker_id = worker_id
        self.job = job

    def run(self, current_time):
        # Schedule job
        new_events = []
        if self.simulation.job_split == "PER_TASK":
            new_events = self.simulation.workers[self.worker_id].schedule_job_heft(
                current_time, self.job)
        # elif self.simulation.job_split == "PER_JOB":
        #     new_events = self.simulation.schedule_job_and_send_job(
        #         self.job, current_time)
        return new_events

    def to_string(self):
        return "[Job Arrival at Worker (Job {})] ++".format(self.job.id)


# for PER_TASK scheduler
class TaskArrival(Event):
    """ Event to signify a TASK arriving at a WORKER. """

    def __init__(self, worker, task, job_id):
        self.worker = worker
        self.task = task
        self.job_id = job_id

    def run(self, current_time):
        # log tracking for this task
        self.task.log.set_task_placed_on_worker_queue_timestamp(current_time)
        return self.worker.add_task(current_time, self.task)

    def to_string(self):
        return "[Task Arrival (Job {} - Task {}) at {}] ---".format(self.job_id, self.task.task_id, self.worker)


class InterResultArrival(Event):
    """ Event to signify a TASK arriving at a WORKER. (?) """

    def __init__(self, worker, prev_task, cur_task):
        self.worker = worker
        self.prev_task = prev_task
        self.cur_task = cur_task

    def run(self, current_time):
        self.cur_task.log.set_task_arrival_at_worker_buffer_timestamp(
            current_time)
        return self.worker.receive_intermediate_result(current_time, self.prev_task, self.cur_task)

    def to_string(self):
        return "[Intermediate Results Arrival]: worker:" + str(self.worker.worker_id) + ", prev_task_id:" + str(self.prev_task.task_id) + ", cur_task_id:" + str(self.cur_task.task_id)


class BatchStartEvent(Event):
    """ Event to signify that a BATCH has been started by the WORKER. """

    def __init__(self, worker, model, job_ids=[], task_type=(-1, -1)):
        self.worker = worker
        self.model = model
        self.job_ids = job_ids    # integers representing the job_ids
        self.task_type = task_type # (workflow_id, task_id)

    def run(self, current_time):
        return []

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch Start (Task {self.task_type}, Jobs {jobs}) at Worker {self.worker.worker_id}]"


class BatchEndEvent(Event):
    """ Event to signify that a BATCH has been performed by the WORKER. """

    def __init__(self, worker, model, job_ids=[], task_type=(-1, -1)):
        self.worker = worker
        self.model = model
        self.job_ids = job_ids    # integers representing the job_ids
        self.task_type = task_type # (workflow_id, task_id)

    def run(self, current_time):
        return self.worker.free_slot(current_time, self.model, self.task_type)

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch End (Task {self.task_type}, Jobs {jobs}) at Worker {self.worker.worker_id}]"


# for PER_JOB scheduler
class JobAssignEvent(Event):
    """
    Used in PER_JOB scheduler.
    Event to signify that a JOB has been assigned to a worker for execution.
    JobArrivalAtScheduler delay generate one another in a chain reaction.
    """

    def __init__(self, worker, job):
        self.worker = worker
        self.job = job

    def run(self, current_time):
        return self.worker.add_job(current_time, self.job)

    def to_string(self):
        return "[Job Assign] ---"


class JobEndEvent(Event):
    """ Event to signify that a JOB has been executed by the NODE.
    JobArrivalAtScheduler delay generate one another in a chain reaction."""

    def __init__(self, worker, job):
        self.worker = worker
        self.job = job

    def run(self, current_time):
        return self.worker.free_slot(current_time, self.job)

    def to_string(self):
        return "[Job End] ==="

class EventOrders:
    """
    Used so that the Simulation keeps track of the priority queue order
    """

    def __init__(self, current_time, event):
        self.priority = current_time
        self.current_time = current_time
        self.event = event

    def __lt__(self, other):
        return self.priority < other.priority

    def to_string(self):
        return ""+str(self.current_time) + " " + self.event.to_string()
