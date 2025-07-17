import sys
import json


if __name__ == "__main__":
    configs_path = sys.argv[1]
    configs = json.loads(open(configs_path, "r").read())

    for config in configs["CONFIGS"]:
        formatted_jobs_per_wf = { i: val for i, val in enumerate(config["TOTAL_NUM_OF_JOBS_PER_WORKFLOW"]) }
        formatted_send_rates = { i: val for i, val in enumerate(config["SEND_RATES_BY_WORKFLOW"]) }
        formatted_alloc = [ (int(k), v) for mig in config["CUSTOM_ALLOCATION"] for k, v in mig.items() ]

        print(config["SCHEDULER_TYPE"])
        print(config["OUT_PATH"])
        print(config["TOTAL_NUM_OF_NODES"])
        print(formatted_jobs_per_wf)
        print(formatted_send_rates)
        print(config["FLEX_LAMBDA"])
        print(config["HERD_K"])
        print(config["HERD_PERIODICITY"] if config["HERD_PERIODICITY"] >= 0 else "np.inf")
        print(config["ENABLE_DYNAMIC_MODEL_LOADING"])
        print(config["ALLOCATION_STRATEGY"])
        print(formatted_alloc)
        print([ k for k,v in formatted_jobs_per_wf.items() if v > 0 ])
        print()