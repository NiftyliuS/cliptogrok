from argparse import ArgumentParser

from train import main

# Default setup - no norm no clip ~35k steps to 95% val
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="all-mod",
                        choices=["add-p97", "sub-p97", "mul-p97", "div-p97", "all-mod", "S5"])
    parser.add_argument("--budget", type=int, default=5e3)  # 5_000 steps
    parser.add_argument("--batch_size", type=int, default=2048)  # 4 times the data, 4 times the batch size
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # Model configuration
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # Optimizer controls
    parser.add_argument("--optimizer", default="Lion")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.97)

    # Seed options
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Clip controls
    parser.add_argument("--init_pattern", type=str, default="all", choices=["all", "edge", "edge_ln"])
    parser.add_argument("--init_norm", type=float, default=1.75)  # 0 = disable
    parser.add_argument("--max_norm", type=float, default=1.75)  # 0 = disable

    parser.add_argument("--plot_progress", action="store_true")
    args = parser.parse_args()
    main(args)
