from core.workflow import *
from core.batch import Batch
from schedulers.algo.herd_algo import *

import pandas as pd


class ShepherdState:

    _batch_counter = 0
    allocation_log = {}

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

        # for round-robin worker ordering
        self.next_worker_idxs = [0 for _ in self.worker_groups]

    def update_batch_counter(self):
        ShepherdState._batch_counter += 1
    
    def worker_completed_batch(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] == batch)
        self.worker_states[worker_id] = None

    def worker_rejected_batch(self, worker_id: int, batch: Batch, current_worker_batch: Batch):
        # if alr. scheduled a different batch, do NOT update state
        if self.worker_states[worker_id] != batch:
            return
        self.worker_states[worker_id] = current_worker_batch

    def preempt_batch_on_worker(self, worker_id: int, new_batch: Batch):
        assert(self.worker_states[worker_id] is not None)
        self.worker_states[worker_id] = new_batch

    def assign_batch_to_worker(self, worker_id: int, batch: Batch):
        assert(self.worker_states[worker_id] is None)
        self.worker_states[worker_id] = batch

    # def __str__(self):
    #     s = ""
    #     for i, group in enumerate(self.worker_groups):
    #         s += f"Group {i} contains workers {[w.worker_id for w in group]}\n"
        
    #     s += "\n"

    #     return s

        #  self.worker_groups = worker_groups
        # self.task_type_to_group = task_type_to_group
        # self.group_task_types = [[tt for tt, gid in self.task_type_to_group.items() if gid==group] 
        #                          for group in range(len(self.worker_groups))]
        # self.task_type_to_model = { tt: get_model_id_for_task_type(tt) for tt in self.task_type_to_group.keys() }
        # self.group_models = [set(self.task_type_to_model[tt] for tt in gtts) for gtts in self.group_task_types]
