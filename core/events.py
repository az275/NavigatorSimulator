from core.job import *
from core.network import *
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
            for task in self.job.tasks:
                task.log.task_arrival_at_scheduler_timestamp = current_time
            new_events = self.simulation.schedule_job_and_send_tasks(
                self.job, current_time)
        # elif self.simulation.job_split == "PER_JOB":
        #     new_events = self.simulation.schedule_job_and_send_job(
        #         self.job, current_time)
        return new_events

    def to_string(self):
        return "[Job Arrival at Scheduler (Job {})] ++".format(self.job.id)
    

class TasksArrivalAtScheduler(Event):
    """
    Event signifying that Task(s) arrived at a Centralized scheduler.
    Only for Centralized Schedulers
    """

    def __init__(self, simulation, tasks):
        assert(len(tasks) > 0)
        self.simulation = simulation
        self.tasks = tasks

    def run(self, current_time):
        for task in self.tasks:
            # only set if not set already (avoid changing order for preempted tasks)
            if task.log.task_arrival_at_scheduler_timestamp == 0:
                task.log.task_arrival_at_scheduler_timestamp = current_time
        return self.simulation.schedule_tasks_on_arrival(self.tasks, current_time)

    def to_string(self):
        return f"[Tasks Arrival at Scheduler (Type: {self.tasks[0].task_type}, Job IDs: {list(map(lambda t: t.job_id, self.tasks))})] ++"


class BatchRejectionAtWorker(Event):
    """
    Event signifying that a worker was busy when a batch was sent for execution, and
    the batch has been sent back to the Centralized scheduler for rescheduling.
    """

    def __init__(self, simulation, worker, batch):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch

    def run(self, current_time):
        assert self.simulation.state.worker_states[self.worker.worker_id].id == self.batch.id
        self.simulation.state.worker_rejected_batch(self.worker.worker_id, self.batch)
        return [EventOrders(current_time, TasksArrivalAtScheduler(self.simulation, self.batch.tasks))] # reschedule batch

    def to_string(self):
        return f"[Batch {self.batch.id} Sent Back by Worker {self.worker.worker_id}]"


class BatchArrivalAtWorker(Event):
    """
    Event signifying that a batch of tasks arrived for execution at a worker.
    """

    def __init__(self, simulation, worker, batch):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch

    def run(self, current_time):
        # NOTE: Sends back tasks if busy (shepherd) or doesn't have model (static heft)  
        if (self.simulation.simulation_name != "shepherd" and not self.worker.GPU_state.does_have_idle_copy(self.batch.model, current_time)) or \
            (self.simulation.simulation_name == "shepherd" and any(s.reserved_batch for s in self.worker.GPU_state.state_at(current_time))):
            return [EventOrders(current_time + CPU_to_CPU_delay(self.batch.size()*self.batch.tasks[0].input_size), 
                                BatchRejectionAtWorker(self.simulation, self.worker, self.batch))]
        for task in self.batch.tasks:
            task.log.set_task_placed_on_worker_queue_timestamp(current_time)
        return self.worker.maybe_start_batch(self.batch, current_time)

    def to_string(self):
        return f"[Batch {self.batch.id} Arrival at Worker {self.worker.worker_id} (Type: {self.batch.tasks[0].task_type}, Job IDs: {self.batch.job_ids})] ++"


class BatchPreemptionAtWorker(Event):
    """
    Event signifying that a batch should be preempted at a worker.
    """

    def __init__(self, simulation, worker, batch, old_batch_id):
        self.simulation = simulation
        self.worker = worker
        self.batch = batch # replacement batch
        self.old_batch_id = old_batch_id # preempted batch

    def run(self, current_time):
        # check if batch to be preempted still exists/is actively executing
        if any(s.reserved_batch and s.reserved_batch.id == self.old_batch_id 
               for s in self.worker.GPU_state.state_at(current_time)):
            for task in self.batch.tasks:
                task.log.set_task_placed_on_worker_queue_timestamp(current_time)
            return self.worker.preempt_batch(self.old_batch_id, self.batch, current_time)
        else:
            # if outdated decision, send back tasks for rescheduling
            return [EventOrders(
                current_time + CPU_to_CPU_delay(self.batch.size()*self.batch.tasks[0].input_size),
                BatchRejectionAtWorker(self.simulation, self.worker, self.batch))]

    def to_string(self):
        return f"[Batch Preemption at Worker {self.worker.worker_id} (Batch {self.old_batch_id} preempted)]"


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

    def __init__(self, worker, batch_id=-1, job_ids=[], task_type=(-1, -1)):
        self.worker = worker
        self.batch_id = batch_id
        self.job_ids = job_ids    # integers representing the job_ids
        self.task_type = task_type # (workflow_id, task_id)

    def run(self, current_time):
        return []

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch {self.batch_id} Start (Task {self.task_type}, Jobs {jobs}) at Worker {self.worker.worker_id}]"


class BatchEndEvent(Event):
    """ Event to signify that a BATCH has been performed by the WORKER. """

    def __init__(self, worker, batch, job_ids=[], task_type=(-1, -1)):
        self.worker = worker
        self.batch = batch
        self.job_ids = job_ids    # integers representing the job_ids
        self.task_type = task_type # (workflow_id, task_id)

    def run(self, current_time):
        if self.worker.did_abandon_batch(self.batch.id):
            return []
        return self.worker.free_slot(current_time, self.batch, self.task_type)

    def to_string(self):
        jobs = ",".join([str(id) for id in self.job_ids])
        return f"[Batch {self.batch.id} End (Task {self.task_type}, Jobs {jobs}) at Worker {self.worker.worker_id}]"


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
