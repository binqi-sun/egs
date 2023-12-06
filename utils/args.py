import argparse

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--model',
        default='models/pretrained',
        help='The Pretrained Model dir'
    )
    argparser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='ID of GPU'
    )
    argparser.add_argument(
        '--in_dot',
        default='data/in_dag.dot',
        help='The input dot file'
    )
    argparser.add_argument(
        '--out_dot',
        default='data/out_dag.dot',
        help='The output dot file'
    )
    argparser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='The number of workers'
    )
    args = argparser.parse_args()
    return args
