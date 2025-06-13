"""  --------       Workflow Parameters     --------  """
# https://keras.io/api/applications/

WORKFLOW_LIST = [
    {"JOB_TYPE": 0,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "textvision",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 51.7,
     "TASKS": [{"MODEL_NAME": "",
                "MODEL_ID": -1,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1, 2],
                "MODEL_SIZE": 0,             # in KB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 1,
                "EXECUTION_TIME": 1,         # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16],
                "BATCH_EXEC_TIME": [1, 1, 1, 1, 1]
                },
               {"MODEL_NAME": "text_encoder",
                "MODEL_ID": 0,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 5677000,       # in kB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 10,        # avg time, in ms
                "MAX_BATCH_SIZE": 6,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 6],
                "BATCH_EXEC_TIME": [10, 10, 10, 11]
                },
               {"MODEL_NAME": "vision_encoder",
                "MODEL_ID": 1,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 20919000,      # in kB
                "INPUT_SIZE": 1000,
                "OUTPUT_SIZE": 1000,
                "EXECUTION_TIME": 31,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 10,         # ms
                "BATCH_SIZES": [1, 4, 8, 16],
                "BATCH_EXEC_TIME": [31, 98, 183, 349]
                },
               {"MODEL_NAME": "flmr",
                "MODEL_ID": 2,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [4],
                "MODEL_SIZE": 854000,        # in KB
                "INPUT_SIZE": 1002,
                "OUTPUT_SIZE": 10,
                "EXECUTION_TIME": 1.7,       # in ms
                "MAX_BATCH_SIZE": 32,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16, 32],
                "BATCH_EXEC_TIME": [1.7, 1.9, 1.9, 2, 2.6, 3.1]
                },
               {"MODEL_NAME": "search",
                "MODEL_ID": 3,
                "TASK_INDEX": 4,
                "PREV_TASK_INDEX": [3],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 777000,        # in KB
                "INPUT_SIZE": 10,
                "OUTPUT_SIZE": 10,
                "EXECUTION_TIME": 18,        # in ms
                "MAX_BATCH_SIZE": 32,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 4, 8, 16, 32],
                "BATCH_EXEC_TIME": [18, 64, 114, 209, 418]
                }
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
                "MODEL_SIZE": 10525000,      # in kB
                "INPUT_SIZE": 1000,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 66,        # avg time, in ms
                "MAX_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 6, 8, 10],
                "BATCH_EXEC_TIME": [66, 68, 70, 73, 76, 90]
                },
               {"MODEL_NAME": "encode_search",
                "MODEL_ID": 5,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [2,3],
                "MODEL_SIZE": 1210000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 17.4,      # in ms
                "MAX_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 6, 8, 10],
                "BATCH_EXEC_TIME": [17.4, 18.4, 18.4, 18.4, 19.5, 19.5]
                },
               {"MODEL_NAME": "text_check",
                "MODEL_ID": 6,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [1],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 7383000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 17,        # in ms
                "MAX_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 6, 8, 10],
                "BATCH_EXEC_TIME": [17, 25, 45, 67.5, 90, 112.5]
                },
               {"MODEL_NAME": "aggregate",
                "MODEL_ID": 7,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 0,             # in kB
                "INPUT_SIZE": 4,
                "OUTPUT_SIZE": 4,
                "EXECUTION_TIME": 1,         # in ms
                "MAX_BATCH_SIZE": 10,
                "MAX_WAIT_TIME": 1,          # ms
                "BATCH_SIZES": [1, 2, 4, 6, 8, 10],
                "BATCH_EXEC_TIME": [1, 1, 1, 1, 1, 1]
                }
               ]
     }
]
