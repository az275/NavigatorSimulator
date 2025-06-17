from pickle import NONE
from core.config import *
from core.network import *
from core.config import *

import pandas as pd

from workers.model_state import *


class Worker(object):
    """ Abstract class representing workers. """

    def __init__(self, simulation, num_free_slots, worker_id):
        self.worker_id = worker_id
        self.simulation = simulation
        self.num_free_slots = num_free_slots
        self.current_batch = [] # track the currently executing batch (if any)
        self.GPU_memory_models = []
        self.GPU_state = GPUState()

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

    #  ----------  LOCAL MEMORY MANAGEMENT AND RETRIEVE  ----------"""
    def fetch_model(self, model, current_time):
        if model == None or self.GPU_state.does_have_idle_copy(model, current_time):
            return 0
        
        fetch_time = 0
        fetch_time = SameMachineCPUtoGPU_delay(model.model_size)

        self.simulation.metadata_service.add_model_cached_location(
            model, self.worker_id, current_time + fetch_time)
        self.GPU_state.fetch_model(model, current_time, fetch_time)
        
        self.model_history_log.loc[len(self.model_history_log)] = {
            "start_time": current_time,
            "end_time": current_time + fetch_time, 
            "model_id": model.model_id,
            "placed_or_evicted": "placed"
        }

        return fetch_time
    
    # NOTE: REQUIRED OVERRIDE
    def get_next_models(self, lookahead_count: int, current_time: float, info_staleness=0):
        """
            Returns a list of up to lookahead_count models in order of when they are
            expected to be executed.
        """
        return []

    def _evict_models_from_GPU(self, models_to_evict, current_time):
        # NOTE: Assumes any number of models can be evicted concurrently!
        eviction_duration = 0
        for model in models_to_evict:
            self.simulation.metadata_service.rm_model_cached_location(
                model, self.worker_id, current_time)
            
            evict_time = SameMachineGPUtoCPU_delay(model.model_size)
            self.GPU_state.evict_model(model, current_time, evict_time)
            eviction_duration = max(evict_time, eviction_duration)

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
            Assumes batches run in earliest task arrival order.
        """
        curr_memory = self.GPU_state.available_memory(current_time)
       
        placed_model_states = self.GPU_state.placed_model_states(current_time)
        if policy == self.LOOKAHEAD_EVICTION:
            next_models = self.get_next_models(3, current_time)
            placed_model_states = sorted(
                placed_model_states, 
                key=lambda m: next_models.index(m.model) if m.model in next_models else len(next_models),
                reverse=True
            )

        models_to_evict = []
        for state in placed_model_states:
            if not state.is_reserved_for_batch:
                curr_memory += state.model.model_size
                models_to_evict.append(state.model)
                if curr_memory >= min_required_memory:
                    return self._evict_models_from_GPU(models_to_evict, current_time)

        return 0

    # ------------------------- cached model history update helper functions ---------------
    def get_history(self, history, current_time, info_staleness) -> list:
        delayed_time = current_time - info_staleness
        last_index = len(history) - 1
        while last_index >= 0:
            if history[last_index][0] <= delayed_time:
                return history[last_index][1].copy()
            last_index -= 1  # check the previous one
        return []