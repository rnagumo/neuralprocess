
# Run training

# Kwargs
export MODEL_NAME=${1:-cnp}

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${MODEL_NAME}

# Config for training
export CONFIG_PATH=./examples/config_1d.json

python3 ./examples/train.py --model ${MODEL_NAME} --seed 0 \
    --max-epochs 100000 --log-interval 10000
