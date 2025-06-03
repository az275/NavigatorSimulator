"""  --------       Workflow Parameters     --------  """
# https://keras.io/api/applications/

WORKFLOW_LIST = [
    {"JOB_TYPE": 0,         # ID of the type of workflow (dependency graph)
     "JOB_NAME": "translation",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 1365,
     "TASKS": [{"MODEL_NAME": "OPT",
                "MODEL_ID": 0,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1,2,3],
                "MODEL_SIZE": 5720000,       # in kB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 561,       # avg time, in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "marian",
                "MODEL_ID": 1,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [4],
                "MODEL_SIZE": 800000,        # in kB
                "INPUT_SIZE": 2,           
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 441,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "mt5",
                "MODEL_ID": 2,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [4],
                "MODEL_SIZE": 2000000,       # in KB
                "INPUT_SIZE": 2,           
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 778,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
                {"MODEL_NAME": "mt5",
                "MODEL_ID": 2,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [4],
                "MODEL_SIZE": 2000000,       # in KB
                "INPUT_SIZE": 2,           
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 803,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
                {"MODEL_NAME": "",
                "MODEL_ID": -1,
                "TASK_INDEX": 4,
                "PREV_TASK_INDEX": [1,2,3],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 0,             # in KB
                "INPUT_SIZE": 2,           
                "OUTPUT_SIZE": 2,
                "EXECUTION_TIME": 1,         # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               ]
     },

    {"JOB_TYPE": 1,
     "JOB_NAME": "question_answer",
     # the minimum amount of time necessary to execute the whole job
     "BEST_EXEC_TIME": 587,
     "TASKS": [{"MODEL_NAME": "OPT",
                "MODEL_ID": 0,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1],
                "MODEL_SIZE": 5720000,       # in kB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 2,            # in kB
                "EXECUTION_TIME": 560,       # avg time, in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "NLI",
                "MODEL_ID": 3,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": 2140000,       # in kB
                "INPUT_SIZE": 1,
                "OUTPUT_SIZE": 1,
                "EXECUTION_TIME": 27,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                }
               ]
     },

    {"JOB_TYPE": 2,  # ID of the type of workflow (dependency graph)
     "JOB_NAME": "img_to_sound",
     "BEST_EXEC_TIME": 359.2,
     "TASKS": [{"MODEL_NAME": "vit",
                "MODEL_ID": 4,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1,2],
                "MODEL_SIZE": 1700000,       # in kB
                "INPUT_SIZE": 3000,          # 224 x 224 x 3 shape, assuming 64 bits representation
                "OUTPUT_SIZE": 20,
                "EXECUTION_TIME": 283,       # avg time, in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "NLI",
                "MODEL_ID": 3,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 2140000,       # in kB
                "INPUT_SIZE": 20,            # 299×299, assuming 64 bits representation
                "OUTPUT_SIZE": 10, 
                "EXECUTION_TIME": 26,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "txt2speech",
                "MODEL_ID": 5,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 2700000,       # in kB
                "INPUT_SIZE": 20,
                "OUTPUT_SIZE": 3000,
                "EXECUTION_TIME": 76,        # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
                {"MODEL_NAME": "aggregate",
                "MODEL_ID": -1,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": -1,            # in kB
                "INPUT_SIZE": 3000,
                "OUTPUT_SIZE": 3000,
                "EXECUTION_TIME": 0.2,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                }
               ]
     },

    {"JOB_TYPE": 3,  # ID of the type of workflow (dependency graph)
     "JOB_NAME": "ImageObjDetect",
     "BEST_EXEC_TIME": 282.6,
     "TASKS": [{"MODEL_NAME": "entry",
                "MODEL_ID": -1,
                "TASK_INDEX": 0,
                "PREV_TASK_INDEX": [],
                "NEXT_TASK_INDEX": [1,2],
                "MODEL_SIZE": -1,            # in kB
                "INPUT_SIZE": 3000,
                "OUTPUT_SIZE": 3000,
                "EXECUTION_TIME": 0.6,       # avg time, in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "DETR",
                "MODEL_ID": 8,
                "TASK_INDEX": 1,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 1800000,       # in kB
                "INPUT_SIZE": 3000,          # 299×299, assuming 64 bits representation
                "OUTPUT_SIZE": 3000,
                "EXECUTION_TIME": 178,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
               {"MODEL_NAME": "Depth",
                "MODEL_ID": 9,
                "TASK_INDEX": 2,
                "PREV_TASK_INDEX": [0],
                "NEXT_TASK_INDEX": [3],
                "MODEL_SIZE": 3900000,       # in kB
                "INPUT_SIZE": 3000,
                "OUTPUT_SIZE": 3000,
                "EXECUTION_TIME": 147,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                },
                {"MODEL_NAME": "Aggregate",
                "MODEL_ID": -1,
                "TASK_INDEX": 3,
                "PREV_TASK_INDEX": [1,2],
                "NEXT_TASK_INDEX": [],
                "MODEL_SIZE": -1,            # in kB
                "INPUT_SIZE": 3000,
                "OUTPUT_SIZE": 3000,
                "EXECUTION_TIME": 104,       # in ms
                "MAX_BATCH_SIZE": 16,
                "MAX_WAIT_TIME": 5000,       # ms
                "SLOWDOWN_FACTOR": 1.2       # batch execution time
                }
               ]
     },

]
