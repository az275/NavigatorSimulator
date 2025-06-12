from pickle import NONE
from core.config import *
from core.network import *
from core.config import *

import pandas as pd


class Worker(object):
    """ Abstract class representing workers. """

    def __init__(self, simulation, num_free_slots, worker_id):
        self.worker_id = worker_id
        self.simulation = simulation
        self.num_free_slots = num_free_slots
        self.current_batch = [] # track the currently executing batch (if any)
        self.GPU_memory_models = []
        # Keep track of the list of models sitting in GPU memory at time: 
        # {time-> list of model objects} : [ (time1,[model0,model1,]), (time2,[model1,...]),...]
        self.GPU_memory_models_history = []
        self.models_in_use = [] # models in use by a currently executing batch

        self.model_history_log = pd.DataFrame(columns=["start_time", "end_time",
                                                       "model_id", "placed_or_evicted"])

    def __hash__(self):
        return hash(self.worker_id)

    def __str__(self):
        return "[Worker_id:{}]".format(self.worker_id)

    def __eq__(self, other):
        if isinstance(other, Worker):
            return self.worker_id == other.worker_id
        return False

    def __ne__(self, other):
        return not (self == other)
    
    def __lt__(self, other):
        return self.worker_id < other.worker_id

    def initial_model_placement(self, model):
        """
        Place the model according to the placement policy
        """
        if (self.used_GPUmemory(0, 0) + model.model_size) < GPU_MEMORY_SIZE:
            self.GPU_memory_models.append(model)
            self.simulation.metadata_service.add_model_cached_location(
                model, self.worker_id, 0)
            return 1
        return 1

    def used_GPUmemory(self, current_time, info_staleness=0, requiring_worker_id=None) -> int:
        """
        Helper function for local GPU memory usage check
        """
        if requiring_worker_id == self.worker_id:
            info_staleness = 0
        models = self.get_model_history(current_time, info_staleness)
        return sum(m.model_size for m in models)

    #  ----------  LOCAL MEMORY MANAGEMENT AND RETRIEVE  ----------"""
    def does_have_model(self, model, current_time: float, info_staleness=0) -> bool:
        w_models = self.get_model_history(current_time, info_staleness)
        return model in w_models
    
    def copies_in_memory(self, model, current_time: float, info_staleness=0) -> int:
        w_models = self.get_model_history(current_time, info_staleness)
        return w_models.count(model)

    def can_fit(self, min_required_memory: int, current_time: float, info_staleness=0) -> bool:
        # models currently being fetched = models in use - models loaded on GPU
        loaded_models = self.get_model_history(current_time, info_staleness)
        fetching_models = []
        for model in self.models_in_use:
            if model in loaded_models:
                loaded_models.remove(model)
            else:
                fetching_models.append(model)

        # loaded models + models currently being fetched
        used_memory = self.used_GPUmemory(current_time, info_staleness=info_staleness) + \
                      sum([model.model_size for model in fetching_models])
        
        # if currently available memory >= min_required_memory
        if GPU_MEMORY_SIZE - used_memory >= min_required_memory:
            return True
        
        # if no batches/current batches do not use GPU
        if self.models_in_use == [] and min_required_memory <= GPU_MEMORY_SIZE:
            return True
        
        # if evicting all except current required models & models being fetched can make enough space
        if GPU_MEMORY_SIZE - sum(map(lambda m: m.model_size, self.models_in_use)) - \
            sum(map(lambda m: m.model_size, fetching_models)) >= min_required_memory:
            return True
        
        return False

    def fetch_model(self, model, current_time):
        """
        Return: model transfer time required to execute the Task
        Every "task" requires one "model" to be executed correctly
        add this information to 2 histories:  
            1. model_history on worker
            2. cache_history on metadata_service
        """
        # check if exists a copy of the model not currently in use
        if model is None or \
            self.copies_in_memory(model, current_time) - self.models_in_use.count(model) > 0:
            return 0
        
        fetch_time = 0
        fetch_time = SameMachineCPUtoGPU_delay(model.model_size)

        self.model_history_log.loc[len(self.model_history_log)] = {
            "start_time": current_time,
            "end_time": current_time + fetch_time, 
            "model_id": model.model_id,
            "placed_or_evicted": "placed"
        }

        self.simulation.metadata_service.add_model_cached_location(
            model, self.worker_id, current_time + fetch_time)
        self.add_model_to_memory_history(model, current_time + fetch_time)
        return fetch_time
    
    # NOTE: REQUIRED OVERRIDE
    def get_next_models(self, lookahead_count: int, current_time: float, info_staleness=0):
        """
            Returns a list of up to lookahead_count models in order of when they are
            expected to be executed.
        """
        return []

    def _evict_models_from_GPU(self, models_to_evict, current_time):
        eviction_duration = 0
        for model in models_to_evict:
            if model not in self.models_in_use:
                self.simulation.metadata_service.rm_model_cached_location(
                    model, self.worker_id, current_time)
                self.rm_model_in_memory_history(model, current_time)
                eviction_duration += SameMachineGPUtoCPU_delay(model.model_size)

                self.model_history_log.loc[len(self.model_history_log)] = {
                    "start_time": current_time,
                    "end_time": current_time + eviction_duration, 
                    "model_id": model.model_id,
                    "placed_or_evicted": "evicted"
                }
        return eviction_duration

    LOOKAHEAD_EVICTION = 0
    FCFS_EVICTION = 1

    def evict_models_from_GPU_until(self, current_time: float, min_required_memory: int, policy: int) -> float:
        """
            Evicts models from GPU according to FCFS or lookahead eviction policy until at least
            min_required_memory space is available. Returns time taken to execute model
            evictions. 0 if min_required_memory could not be created.
            Assumes batches run in first task arrival order.
        """
        if not self.can_fit(min_required_memory, current_time):
            return 0
        
        curr_memory = GPU_MEMORY_SIZE - self.used_GPUmemory(current_time)
       
        models_in_GPU = self.get_model_history(current_time, info_staleness=0)
        if policy == self.LOOKAHEAD_EVICTION:
            next_models = self.get_next_models(3, current_time)
            models_in_GPU = sorted(
                models_in_GPU, 
                key=lambda m: next_models.index(m) if m in next_models else len(next_models),
                reverse=True
            )

        models_to_evict = []
        for model in models_in_GPU:
            if model not in self.models_in_use:
                curr_memory -= model.model_size
                models_to_evict.append(model)
                if curr_memory >= min_required_memory:
                    return self._evict_models_from_GPU(models_to_evict, current_time)
        
        return 0

    # ------------------------- cached model history update helper functions ---------------
    def add_model_to_memory_history(self, model, current_time):
        assert (model.model_size <= GPU_MEMORY_SIZE)
        last_index = len(self.GPU_memory_models_history) - 1
        # 0. base case
        if last_index == -1:
            self.GPU_memory_models_history.append((current_time, [model]))
            return
        # 1. Find the time_stamp place to add this queue information
        while last_index >= 0:
            if self.GPU_memory_models_history[last_index][0] == current_time:
                if model not in self.GPU_memory_models_history[last_index][1]:
                    self.GPU_memory_models_history[last_index][1].append(model)
                break
            if self.GPU_memory_models_history[last_index][0] < current_time:
                if model not in self.GPU_memory_models_history[last_index][1]:
                    next_queue = self.GPU_memory_models_history[last_index][1].copy(
                    )
                    next_queue.append(model)
                    last_index += 1
                    self.GPU_memory_models_history.insert(
                        last_index, (current_time, next_queue)
                    )
                break
            # check the previous entry
            last_index -= 1
        # 2. added the worker_id to all the subsequent timestamp tuples
        while last_index < len(self.GPU_memory_models_history):
            if model not in self.GPU_memory_models_history[last_index][1]:
                self.GPU_memory_models_history[last_index][1].append(model)
            last_index += 1

    def rm_model_in_memory_history(self, model, current_time):
        last_index = len(self.GPU_memory_models_history) - 1
        # 0. base case: shouldn't happen
        if last_index == -1:
            AssertionError("rm model cached location to an empty list")
            return
        # 1. find the place to add this remove_event to the tuple list
        while last_index >= 0:
            if self.GPU_memory_models_history[last_index][0] == current_time:
                if model in self.GPU_memory_models_history[last_index][1]:
                    self.GPU_memory_models_history[last_index][1].remove(model)
                break
            if self.GPU_memory_models_history[last_index][0] < current_time:
                if model in self.GPU_memory_models_history[last_index][1]:
                    next_tasks_in_memory = self.GPU_memory_models_history[last_index][1].copy(
                    )
                    next_tasks_in_memory.remove(model)
                    last_index = last_index + 1
                    self.GPU_memory_models_history.insert(
                        last_index, (current_time, next_tasks_in_memory)
                    )
                break
            last_index -= 1  # go to prev time
        # 2. remove the task from all the subsequent tuple
        while last_index < len(self.GPU_memory_models_history):
            if model in self.GPU_memory_models_history[last_index]:
                self.GPU_memory_models_history[last_index][1].remove(model)
            last_index += 1  # do this for the remaining element after

    def get_history(self, history, current_time, info_staleness) -> list:
        delayed_time = current_time - info_staleness
        last_index = len(history) - 1
        while last_index >= 0:
            if history[last_index][0] <= delayed_time:
                return history[last_index][1].copy()
            last_index -= 1  # check the previous one
        return []

    def get_model_history(self, current_time, info_staleness=0, requiring_workerid= None) -> list:
        if requiring_workerid == self.worker_id:
            info_staleness = 0
        return self.get_history(self.GPU_memory_models_history, current_time, info_staleness)


