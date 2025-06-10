from pickle import NONE
from core.config import *
from core.network import *
from core.config import *
import sys


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
    
    def can_fit(self, min_required_memory: int, current_time: float, info_staleness=0) -> bool:
        # if currently available memory >= min_required_memory
        used_memory = self.used_GPUmemory(current_time, info_staleness=info_staleness)
        if GPU_MEMORY_SIZE - used_memory >= min_required_memory:
            return True
        
        # if not executing any batches or executing a batch with no model,
        # existing models can be evicted to make space
        if (self.current_batch == [] or self.current_batch[0].model == None) and \
            min_required_memory <= GPU_MEMORY_SIZE:
            return True
        
        # if evicting all except current batch's required model can make enough space
        if GPU_MEMORY_SIZE - self.current_batch[0].model.model_size >= min_required_memory:
            return True

    def fetch_model(self, model, current_time):
        """
        Return: model transfer time required to execute the Task
        Every "task" requires one "model" to be executed correctly
        add this information to 2 histories:  
            1. model_history on worker
            2. cache_history on metadata_service
        """
        if model is None:
            return 0
        # First check if the model is stored locally: either on GPU, or systemRAM(home node)
        # case1: if it is in local GPU already
        if self.does_have_model(model, current_time):
            return 0
        fetch_time = 0
        fetch_time = SameMachineCPUtoGPU_delay(model.model_size)
        self.simulation.metadata_service.add_model_cached_location(
            model, self.worker_id, current_time + fetch_time)
        self.add_model_to_memory_history(model, current_time + fetch_time)
        eviction_time = self.evict_model_from_GPU(current_time + fetch_time)
        return fetch_time + eviction_time
    
    # Required to be overriden
    def get_next_tasks(self, lookahead_count: int, current_time: float, info_staleness=0):
        """
            Returns a list of up to lookahead_count tasks in order of when they are
            expected to begin execution on the worker.
        """
        return []


    def _evict_models_from_GPU(self, models_to_evict, current_time):
        eviction_duration = 0
        required_current_model = self.current_batch[0].model if self.current_batch else None
        for model in models_to_evict:
            if model != required_current_model:
                self.simulation.metadata_service.rm_model_cached_location(
                    model, self.worker_id, current_time)
                self.rm_model_in_memory_history(model, current_time)
                eviction_duration += SameMachineGPUtoCPU_delay(model.model_size)
        return eviction_duration


    def evict_models_from_GPU_until(self, current_time: float, min_required_memory: int) -> float:
        """
            Evicts models from GPU according to lookahead eviction policy until at least
            min_required_memory space is available. Returns time taken to execute model
            evictions. 0 if min_required_memory could not be created.
        """
        if not self.can_fit(min_required_memory, current_time):
            return 0
        
        curr_memory = GPU_MEMORY_SIZE - self.used_GPUmemory(current_time)
       
        models_in_GPU = self.get_model_history(current_time, info_staleness=0)
        required_current_model = self.current_batch[0].model if self.current_batch else None
        next_models = set(map(lambda task: task.model, self.get_next_tasks(3)))

        models_to_evict = []

        for i, model in enumerate(models_in_GPU):
            # lowest priority models
            if model not in next_models and model != required_current_model:
                curr_memory -= model.model_size
                models_to_evict.append(model)
                if curr_memory >= min_required_memory:
                    return self._evict_models_from_GPU(models_to_evict)

        # next look at future models from latest -> earliest to be used
        for model in next_models[::-1]:
            if model in models_in_GPU and model != required_current_model:
                curr_memory -= model.model_size
                models_to_evict.append(model)
                if curr_memory >= min_required_memory:
                    return self._evict_models_from_GPU(models_to_evict)
        
        return 0
    

    def evict_model_from_GPU(self, current_time):
        """
        Do nothing if current cached models didn't exceed the GPU memory
        remove this information to 2 histories:  
            1. model_history on worker
            2. cache_history on metadata_service
        """
        models_in_GPU = self.get_model_history(current_time, info_staleness=0)
        models_total_size = 0
        for model in models_in_GPU:
            models_total_size += model.model_size
        eviction_index = 0
        eviction_duration = 0
        while (models_total_size > GPU_MEMORY_SIZE):
            rm_model = models_in_GPU[eviction_index]
            self.simulation.metadata_service.rm_model_cached_location(
                rm_model, self.worker_id, current_time)
            self.rm_model_in_memory_history(rm_model, current_time)
            models_total_size -= rm_model.model_size
            eviction_index += 1
            eviction_duration += SameMachineGPUtoCPU_delay(rm_model.model_size)
        return eviction_duration

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


