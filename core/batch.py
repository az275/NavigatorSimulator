from core.task import Task


class Batch:
    def __init__(self, id: int, tasks: list[Task]):
        assert(len(tasks) > 0)

        self.id = id
        self.tasks = tasks
        self.model = tasks[0].model
        self.job_ids = list(map(lambda t: t.job_id, tasks))

        self.execution_start_timestamp = -1
        self.front_queue_timestamp = -1

    def size(self) -> int:
        return len(self.tasks)
    
    def __str__(self):
        return f"[BATCH {self.id} | TYPE {self.tasks[0].task_type}] <JOBS {self.job_ids}>"
    
    def __repr__(self):
        return self.__str__()