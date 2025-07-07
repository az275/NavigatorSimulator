"""  --------       Workflow Parameters     --------  """
# https://keras.io/api/applications/

WORKFLOW_LIST = [
    {"JOB_TYPE": 0,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 51.7,
     "TASKS": [
               {"MODEL_NAME": "text_encoder",
                "MODEL_ID": 0,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [2],
                "MODEL_SIZE": 5100000, # 5677000,       # in kB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 10,        # avg time, in ms
                "MAX_BATCH_SIZE": 6,
                "MAX_WAIT_TIME": 1,         # ms
                "BATCH_SIZES": [1, 2, 4, 6],
                "BATCH_EXEC_TIME": [10, 10, 10, 11],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [10, 10, 10, 11],
                    12: [9, 9, 9, 10],
                    6: [7, 7, 7, 8]
                },
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05},
               {"MODEL_NAME": "vision_encoder",
                "MODEL_ID": 1,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [2],
                "MODEL_SIZE": 20919000,      # in kB
                "INPUT_SIZE": 10000,
                "OUTPUT_SIZE": 1000,
                "EXECUTION_TIME": 31,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 10,        # ms
                "BATCH_SIZES": [1, 4, 8, 16],
                "BATCH_EXEC_TIME": [31, 98, 183, 349],
                "MIG_BATCH_EXEC_TIMES": {24: [31, 98, 183, 349]},
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05},
               {"MODEL_NAME": "flmr",
                "MODEL_ID": 2,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0,1],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 854000,        # in KB
                "INPUT_SIZE": 1002,
                "OUTPUT_SIZE": 10,
                "EXECUTION_TIME": 1.7,       # in ms
                "MAX_BATCH_SIZE": 32,
                "MAX_WAIT_TIME": 1,         # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16, 32],
                "BATCH_EXEC_TIME": [1.7, 1.9, 1.9, 2, 2.6, 3.1],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [1.7, 1.9, 1.9, 2, 2.6, 3.1],
                    12: [1.5, 1.7, 1.7, 1.8, 2.3, 2.8],
                    6: [1.3, 1.4, 1.4, 1.5, 1.9, 2.3]
                },
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05},
               {"MODEL_NAME": "search",
                "MODEL_ID": 3,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 777000,        # in KB
                "INPUT_SIZE": 10,
                "OUTPUT_SIZE": 10,
                "EXECUTION_TIME": 18,        # in ms
                "MAX_BATCH_SIZE": 32,
                "MAX_WAIT_TIME": 1,        # ms
                "BATCH_SIZES": [1, 4, 8, 16, 32],
                "BATCH_EXEC_TIME": [18, 64, 114, 209, 418],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [18, 64, 114, 209, 418],
                    12: [14, 50, 90, 164, 328],
                    6: [14, 50, 90, 164, 328]
                },
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05}
               ]
     },
     {"JOB_TYPE": 1,
     "JOB_NAME": "tts",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 101.4,
     "TASKS": [{"MODEL_NAME": "audio_det",
                "MODEL_ID": 4,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1],
                "MODEL_SIZE": 6093000,      # in kB
                "INPUT_SIZE": 1000,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 65,        # avg time, in ms
                "MAX_BATCH_SIZE": 8,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 8],
                "BATCH_EXEC_TIME": [65.0, 68.0, 69.4, 72.1],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [65.0, 68.0, 69.4, 72.1],
                    12: [65.0, 68.0, 69.4, 72.1] # TODO: update with real nums
                },
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05},
               {"MODEL_NAME": "encode_search-ivf",
                "MODEL_ID": 5,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [2,3],
                "MODEL_SIZE": 1210000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 16.7,      # in ms
                "MAX_BATCH_SIZE": 8,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 8],
                "BATCH_EXEC_TIME": [16.7, 17.2, 17.5, 17.5],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [16.7, 17.2, 17.5, 17.5], # TODO [0.397, 0.405, 0.424, 0.456],
                    12: [16.7, 17.2, 17.5, 17.5],
                    6: [16.5, 16.9, 16.9, 17.3]
                },
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05},
               {"MODEL_NAME": "text_check",
                "MODEL_ID": 6,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [1],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 2101000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 3.36,        # in ms
                "MAX_BATCH_SIZE": 2,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2],
                "BATCH_EXEC_TIME": [3.36, 3.72],
                "MIG_BATCH_EXEC_TIMES": {
                    24: [3.36, 3.72], 12: [4.04, 3.97], 6: [5.66, 9.51]
                },
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05},
               {"MODEL_NAME": "aggregate-tts",
                "MODEL_ID": 7,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 6135000,             # in kB
                "INPUT_SIZE": 4,
                "OUTPUT_SIZE": 4,
                "EXECUTION_TIME": 87.3,         # in ms
                "MAX_BATCH_SIZE": 1,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1],
                "BATCH_EXEC_TIME": [87.3],
                "MIG_BATCH_EXEC_TIMES": {24: [87.3], 12: [149.3]},
                "EXEC_TIME_COEFFICIENT_OF_VARIATION": 0.05}
               ]
     }
]
