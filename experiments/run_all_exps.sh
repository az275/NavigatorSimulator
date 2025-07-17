#!/bin/zsh

configs_path=$1
echo "Parsing configs from $configs_path"

root_dir=$(pwd)
configs_out=$(python3 parse_configs_json.py $configs_path)

run_experiment() {
    local cfg_prop_list=("$@")

    sed -i '.bak' -e "s/^plotting_job_type_list = .*$/plotting_job_type_list = ${cfg_prop_list[12]}/" \
        run_experiment.py

    sed -i '.bak' -e "s/^TOTAL_NUM_OF_NODES = .*$/TOTAL_NUM_OF_NODES = ${cfg_prop_list[3]}/" \
        -e "s/^TOTAL_NUM_OF_JOBS_PER_WORKFLOW = .*$/TOTAL_NUM_OF_JOBS_PER_WORKFLOW = ${cfg_prop_list[4]}/" \
        -e "s/^SEND_RATES_BY_WORKFLOW = .*$/SEND_RATES_BY_WORKFLOW = ${cfg_prop_list[5]}/" \
        -e "s/^FLEX_LAMBDA = .*$/FLEX_LAMBDA = ${cfg_prop_list[6]}/" \
        -e "s/^HERD_K = .*$/HERD_K = ${cfg_prop_list[7]}/" \
        -e "s/^HERD_PERIODICITY = .*$/HERD_PERIODICITY = ${cfg_prop_list[8]}/" \
        -e "s/^ENABLE_DYNAMIC_MODEL_LOADING = .*$/ENABLE_DYNAMIC_MODEL_LOADING = ${cfg_prop_list[9]}/" \
        -e "s/^ALLOCATION_STRATEGY = .*$/ALLOCATION_STRATEGY = '${cfg_prop_list[10]}'/" \
        -e "s/^CUSTOM_ALLOCATION = .*$/CUSTOM_ALLOCATION = ${cfg_prop_list[11]}/" \
        ../core/config.py
    
    cat ../core/config.py

    python3 run_experiment.py "${cfg_prop_list[1]}" > log.txt

    mv log.txt results/"${cfg_prop_list[1]}"
    cp ../core/config.py results/"${cfg_prop_list[1]}"
    mv results/"${cfg_prop_list[1]}" $root_dir/results/"${cfg_prop_list[2]}"

    mv ../core/config.py.bak ../core/config.py
    mv run_experiment.py.bak run_experiment.py
}

mkdir -p results

if [ ! -d "exps/1" ]; then
    mkdir -p exps/1

    cd exps/1
    git clone https://github.com/az275/NavigatorSimulator.git
    cd NavigatorSimulator
    git checkout shepherd
    cd $root_dir

    cp -r exps/1 exps/2
    cp -r exps/1 exps/3
    cp -r exps/1 exps/4
    cp -r exps/1 exps/5
fi

prop_list=()
i=0
IFS=$'\n'
while read -r property; do
    if [[ -z "$property" ]]; then
        clone_num=$((i % 5 + 1))
        echo "Current clone: $clone_num"

        if [[ "$clone_num" -eq "0" && "$i" -ne "0" ]]; then
            echo "Waiting for previous to finish..."
            wait
        fi

        cd $root_dir/exps/$clone_num/NavigatorSimulator
        source set_env.sh
        cd experiments

        echo "Running config..."
        run_experiment "${prop_list[@]}" &

        cd $root_dir

        prop_list=()
        ((i++))
    else
        prop_list+=("$property")
    fi
done <<< "$configs_out"

if [ "${#prop_list[@]}" -gt 0 ]; then
    clone_num=$((i % 5 + 1))
    echo "Current clone: $clone_num"

    cd $root_dir/exps/$clone_num/NavigatorSimulator
    source set_env.sh
    cd experiments

    echo "Running config..."
    run_experiment "${prop_list[@]}" &
fi

echo "Started all configs. Waiting for tasks to finish..."
wait