import sys
import os
for mod in list(sys.modules.keys()):
    if mod in ['train_hyper', 'models_hyper', 'dataset_hyper']:
        del sys.modules[mod]

from train_hyper import train_hyper
import argparse
import torch
import numpy as np
import random
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',        type=int,   default=0)
    parser.add_argument('--seed',          type=int,   default=0)
    parser.add_argument('--n_folds',       type=int,   default=10)
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_dim',    type=int,   default=64)
    parser.add_argument('--n_layers',      type=int,   default=2)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--patience',      type=int,   default=50)
    parser.add_argument('--num_classes',   type=int,   default=2)
    parser.add_argument('--dataset',       type=str,   default='adni')
    parser.add_argument('--mode',          type=str,   default='function')
    parser.add_argument('--network',       type=str,   default='yeo7')
    parser.add_argument('--model',        type=str,   default='hgnn', choices=['hgnn', 'transformer'])
    parser.add_argument('--alpha_mode',   type=str,   default='attn+hcof', choices=['attn', 'hcof', 'attn+hcof'])
    parser.add_argument('--roi2net',      type=str,   default=None)   # AAL -> Yeo7 label file
    parser.add_argument('--window',       type=int,   default=20)     # w in hDYN
    parser.add_argument('--edge_density', type=float, default=0.10)
    parser.add_argument('--sca_tau',      type=int,   default=1)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cpu'
    seed   = args.seed

    # ── Determinism ──
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device(f'cuda:{args.gpu_id}')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
          f"\tYeo7 Sequential Transformer | "
          f"device={device} | seed={seed}")

    train_hyper(args, device)


if __name__ == '__main__':
    main()
