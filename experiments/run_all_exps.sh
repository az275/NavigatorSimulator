#!/bin/zsh

configs_path=$1
echo "Parsing configs from $configs_path"

run_experiment() {
    local cfg_prop_list=("$@")

    sed -i '.bak' -e "s/^TOTAL_NUM_OF_NODES = .*$/TOTAL_NUM_OF_NODES = ${cfg_prop_list[2]}/" \
        -e "s/^TOTAL_NUM_OF_JOBS_PER_WORKFLOW = .*$/TOTAL_NUM_OF_JOBS_PER_WORKFLOW = ${cfg_prop_list[3]}/" \
        -e "s/^SEND_RATES_BY_WORKFLOW = .*$/SEND_RATES_BY_WORKFLOW = ${cfg_prop_list[4]}/" \
        -e "s/^FLEX_LAMBDA = .*$/FLEX_LAMBDA = ${cfg_prop_list[5]}/" \
        -e "s/^HERD_K = .*$/HERD_K = ${cfg_prop_list[6]}/" \
        -e "s/^HERD_PERIODICITY = .*$/HERD_PERIODICITY = ${cfg_prop_list[7]}/" \
        -e "s/^ENABLE_DYNAMIC_MODEL_LOADING = .*$/ENABLE_DYNAMIC_MODEL_LOADING = ${cfg_prop_list[8]}/" \
        -e "s/^ALLOCATION_STRATEGY = .*$/ALLOCATION_STRATEGY = '${cfg_prop_list[9]}'/" \
        -e "s/^CUSTOM_ALLOCATION = .*$/CUSTOM_ALLOCATION = ${cfg_prop_list[10]}/" \
        ../core/config.py
    
    python3 run_experiments.py "${cfg_prop_list[0]}" > log.txt

    mv log.txt results/"${cfg_prop_list[0]}"
    cp ../core/config.py results/"${cfg_prop_list[0]}"
    mv results/"${cfg_prop_list[0]}" results/"${cfg_prop_list[1]}"

    echo "Results in results/${cfg_prop_list[1]}"

    mv ../core/config.py.bak ../core/config.py
}

configs_out=$(python3 parse_configs_json.py $configs_path)
prop_list=()
IFS=$'\n'
while read -r property; do
    if [[ -z "$property" ]]; then
        echo "Running config..."
        run_experiment "${prop_list[@]}"
        prop_list=()
    else
        prop_list+=("$property")
    fi
done <<< "$configs_out"

if [ "${#prop_list[@]}" -gt 0 ]; then
    run_experiment "${prop_list[@]}"
fi