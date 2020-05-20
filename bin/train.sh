
# Run training

# Kwargs
export MODEL_NAME=${1:-cnp}
SEED=${2:-0}

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=tmp

# Config for training
export CONFIG_PATH=./examples/config_1d.json

python3 ./examples/run.py --model ${MODEL_NAME} --seed ${SEED} \
    --epochs 200000 --log-save-interval 20000
