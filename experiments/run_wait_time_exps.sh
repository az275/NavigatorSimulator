mkdir -p results/boostcentralheft/wait_time_exps

max_wait_time="0.05" # ms

for i in {1..12}; do
    max_wait_time=$(awk "BEGIN { print $max_wait_time * 2 }")

    sed -i '.bak' "s|max_wait_time|$max_wait_time|g" ../core/workflow.py

    echo "Running for $max_wait_time ms..."

    python3 run_experiments.py boostcentralheft > results/boostcentralheft/wait_time_exps/$max_wait_time.txt

    mv ../core/workflow.py.bak ../core/workflow.py
done