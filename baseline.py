from argparse import ArgumentParser

from train import main

# Default setup - no norm no clip ~35k steps to 95% val
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # Model configuration
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # Optimizer controls
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    # Seed options
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # Clip controls
    parser.add_argument("--init_pattern", type=str, default="all", choices=["all", "edge", "edge_ln"])
    parser.add_argument("--init_norm", type=float, default=0.0)  # 0 = disable
    parser.add_argument("--max_norm", type=float, default=0.0)  # 0 = disable

    args = parser.parse_args()
    main(args)
