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
                "MAX_BATCH_SIZE": 128,
                "MAX_WAIT_TIME": 50,         # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16, 32, 64, 128],
                "BATCH_EXEC_TIME": [1, 1, 1, 1, 1, 1, 1, 1]
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
                "MAX_BATCH_SIZE": 128,
                "MAX_WAIT_TIME": 50,         # ms
                "BATCH_SIZES": [1, 4, 8, 16, 32, 64, 128],
                "BATCH_EXEC_TIME": [10, 10, 11, 12, 15, 20, 31]
                },
               {"MODEL_NAME": "vision_encoder",
                "MODEL_ID": 1,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 11655000,      # in kB
                "INPUT_SIZE": 10000,
                "OUTPUT_SIZE": 100,
                "EXECUTION_TIME": 31,        # in ms
                "MAX_BATCH_SIZE": 8,
                "MAX_WAIT_TIME": 100,        # ms
                "BATCH_SIZES": [1, 4, 8],
                "BATCH_EXEC_TIME": [31, 98, 183]
                },
               {"MODEL_NAME": "flmr",
                "MODEL_ID": 2,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [4],
                "MODEL_SIZE": 854000,        # in KB
                "INPUT_SIZE": 102,
                "OUTPUT_SIZE": 5,
                "EXECUTION_TIME": 1.7,       # in ms
                "MAX_BATCH_SIZE": 32,
                "MAX_WAIT_TIME": 50,         # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16, 32],
                "BATCH_EXEC_TIME": [1.7, 1.9, 1.9, 2, 2.6, 3.1]
                },
               {"MODEL_NAME": "search",
                "MODEL_ID": 3,
                "TASK_INDEX": 4,
                "PREV_TASK_INDEX": [3],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 777000,        # in KB
                "INPUT_SIZE": 5,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 18,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 100,        # ms
                "BATCH_SIZES": [1, 4, 8, 16],
                "BATCH_EXEC_TIME": [18, 64, 114, 209]
                }
               ]
     },

    {"JOB_TYPE": 1,
     "JOB_NAME": "tts",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 308.4,
     "TASKS": [{"MODEL_NAME": "audio_det",
                "MODEL_ID": 4,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1],
                "MODEL_SIZE": 10525000,      # in kB
                "INPUT_SIZE": 10000,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 66,        # avg time, in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 100,        # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16],
                "BATCH_EXEC_TIME": [66, 68, 70, 76, 127]
                },
               {"MODEL_NAME": "text_encoder_2",
                "MODEL_ID": 5,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [2],
                "MODEL_SIZE": 427000,        # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 4,
                "EXECUTION_TIME": 17,        # in ms
                "MAX_BATCH_SIZE": 64,
                "MAX_WAIT_TIME": 50,         # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16, 32, 64],
                "BATCH_EXEC_TIME": [17, 18, 18, 19, 19, 20, 22]
                },
               {"MODEL_NAME": "faiss_search",
                "MODEL_ID": 6,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [1],
                "NEXT_TASK_INDEX": [3,4],
                "MODEL_SIZE": 783000,        # in kB
                "INPUT_SIZE": 4,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 0.4,       # in ms
                "MAX_BATCH_SIZE": 256,
                "MAX_WAIT_TIME": 50,         # ms
                "BATCH_SIZES": [1, 2, 4, 8, 16, 32, 64, 128, 256],
                "BATCH_EXEC_TIME": [0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.8, 1.1, 1.6]
                },
               {"MODEL_NAME": "text_check",
                "MODEL_ID": 7,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [2],
                "NEXT_TASK_INDEX": [4],
                "MODEL_SIZE": 7383000,       # in kB
                "INPUT_SIZE": 2,
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 17,        # in ms
                "MAX_BATCH_SIZE": 4,
                "MAX_WAIT_TIME": 50,         # ms
                "BATCH_SIZES": [1, 2, 4],
                "BATCH_EXEC_TIME": [17, 25, 45]
                },
               {"MODEL_NAME": "text_to_speech",
                "MODEL_ID": 8,
                "TASK_INDEX": 4,
                "PREV_TASK_INDEX": [2,3],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 783000,        # in kB
                "INPUT_SIZE": 4,
                "OUTPUT_SIZE": 10000,
                "EXECUTION_TIME": 208,       # in ms
                "MAX_BATCH_SIZE": 1,
                "MAX_WAIT_TIME": 100,        # ms
                "BATCH_SIZES": [1],
                "BATCH_EXEC_TIME": [208]
                }
               ]
     }
]
