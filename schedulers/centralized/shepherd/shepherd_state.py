from core.workflow import *
from core.batch import Batch
from schedulers.algo.herd_algo import *


class ShepherdState:

    def __init__(self, worker_groups: list[list], task_type_to_group: dict[tuple[int,int],int]):
        self.worker_groups = worker_groups
        self.task_type_to_group = task_type_to_group
        self.group_task_types = [[tt for tt, gid in self.task_type_to_group.items() if gid==group] 
                                 for group in range(len(self.worker_groups))]
        self.task_type_to_model = { tt: get_model_id_for_task_type(tt) for tt in self.task_type_to_group.keys() }
        self.group_models = [set(self.task_type_to_model[tt] for tt in gtts) for gtts in self.group_task_types]
        
        # init currently executing batch ids
        self.worker_states = {}
        for group in self.worker_groups:
            for worker in group:
                self.worker_states[worker.worker_id] = None

        self._batch_counter = 0

    def update_batch_counter(self):
        self._batch_counter += 1
    
    def worker_completed_batch(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] == batch)
        self.worker_states[worker_id] = None

    def worker_rejected_batch(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] == batch)
        self.worker_states[worker_id] = None

    def preempt_batch_on_worker(self, worker_id: int, new_batch: Batch):
        assert(self.worker_states[worker_id] is not None)
        self.worker_states[worker_id] = new_batch

    def assign_batch_to_worker(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] is None)
        self.worker_states[worker_id] = batch
