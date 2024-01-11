cd /home/seb2000/Causal_NF

source LIGHTNING_HYDRA_DEPS/bin/activate
set -a
export PYTHONPATH="$PYTHONPATH:$PWD"

python3 src/train.py
