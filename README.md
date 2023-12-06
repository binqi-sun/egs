# Edge Generation Scheduling

Edge Generation Scheduling (EGS) is an algorithm for DAG scheduling using deep reinforcement learning (DRL).

## How to Run EGS

First, install the required python packages listed in `requirements.txt` using  `pip` or `conda`. Then, run the EGS with:

```
python egs.py
```

Optional arguments:

* `--in_dot`: the path to the input `.dot` file (default: `data/in_dag.dot`).
* `--out_dot`: the path to the output `.dot` file (default: `data/out_dag.dot`).
* `--model`: the directory of the pretrained neural network model (default: `models/pretrained`). If no pretrained model is provided, a random policy will be used instead.
* `--workers`: the number of workers (processors) used to schedule the input DAG task (default: `None`). If the number of workers is not specified, the EGS will return a schedule with the minimum number of workers.
* `--gpu_id`: the ID of the GPU that is going to be used for neural network inference (default: `0`).
