import math
from argparse import ArgumentParser
from lion_pytorch import Lion
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import LlamaModel, LlamaForCausalLM, LlamaConfig

from SignSGD import SignSGD
from goblin_for_causal_lm import GoblinConfig, GoblinForCausalLM
from norms import project_to_sphere, clip_weight_norms
from datasets import (
    addition_mod_p_data,
    subtraction_mod_p_data,
    multiplication_mod_p_data,
    division_mod_p_data,
    permutation_s5_data,
)


def main(args):
    print("Warning!You are testing LLamaForCausalLM / GoblinForCausalLM not the paper models!")
    if args.random_seed:
        args.seed = torch.seed()
    else:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Task-specific data generation and model config
    if args.task == 'add-p97':
        p = 97

        data = addition_mod_p_data(p, eq_token=p, op_token=p + 1)

        num_tokens, seq_len = p + 2, 5
    elif args.task == 'sub-p97':
        p = 97

        data = subtraction_mod_p_data(p, eq_token=p, op_token=p + 1)

        num_tokens, seq_len = p + 2, 5
    elif args.task == 'mul-p97':
        p = 97

        data = multiplication_mod_p_data(p, eq_token=p, op_token=p + 1)

        num_tokens, seq_len = p + 2, 5
    elif args.task == 'div-p97':
        p = 97

        data = division_mod_p_data(p, eq_token=p, op_token=p + 1)

        num_tokens, seq_len = p + 2, 5
    elif args.task == 'S5':

        data = permutation_s5_data(eq_token=120, op_token=121)

        num_tokens, seq_len = 122, 5
    elif args.task == 'all-mod':
        p = 97
        eq_token = 97

        add_data = addition_mod_p_data(p, eq_token=eq_token, op_token=98)
        sub_data = subtraction_mod_p_data(p, eq_token=eq_token, op_token=99)
        mul_data = multiplication_mod_p_data(p, eq_token=eq_token, op_token=100)
        div_data = division_mod_p_data(p, eq_token=eq_token, op_token=101)

        data = torch.cat([add_data, sub_data, mul_data, div_data], dim=1)
        num_tokens, seq_len = 102, 5

    else:
        raise ValueError(f"Unknown task: {args.task}")

    if args.model == "goblin":
        config = GoblinConfig(
            vocab_size=num_tokens,

            hidden_size=args.dim,
            intermediate_size=352,
            num_hidden_layers=args.num_layers,

            attn_heads=4,
            conv_k=2,
        )
        model_name = "GoblinForCausalLM"
        model = GoblinForCausalLM(config).to(device)
    elif args.model == "llama":
        config = LlamaConfig(
            vocab_size=num_tokens,
            max_position_embeddings=seq_len,

            hidden_size=args.dim,
            intermediate_size=352,
            num_hidden_layers=args.num_layers,

            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model_name = "LlamaForCausalLM"
        model = LlamaForCausalLM(config).to(device)
    else:
        print("Please select model=[goblin|llama]")
        exit(1)

    #### Counting model parameters ####
    model_num_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {model_num_params}")
    #### Norm layers to equal magnitude ####
    if args.init_norm > 0:
        with torch.no_grad():
            patterns = {
                'all': ['*'],  # all layers
                'edge': ['embed_tokens', 'lm_head'],  # first + last
            }[args.init_pattern]
            project_to_sphere(model, args.init_norm, patterns)
    ########################################

    ## CLip layers to max_norm magnitude ##
    if args.max_norm > 0:
        with torch.no_grad():
            clip_weight_norms(model, args.max_norm, skip_patterns=['embed_tokens', 'lm_head'])
    # train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2) # original code

    ##### Configurable train to validation data ratio #####
    train_size = int(data.shape[1] * args.train_ratio)
    perm = torch.randperm(data.shape[1])
    train_idx, valid_idx = perm[:train_size], perm[train_size:]
    #######################################################

    train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    ### SignSGD optimizer inclusion ###
    if args.optimizer == 'SignSGD':
        # We don't use betas so we 'nan' them to not be confusing
        args.beta2 = args.beta1 = float('nan')
        optimizer = SignSGD(
            model.parameters(),
            lr=args.lr
        )
    ### Lion optimizer inclusion ###
    elif args.optimizer == 'Lion':
        # For most Lion experiments we used optimizer with learning rate 1e-3 to 1e-4,
        # weight decay 0, β1 = 0.9, β2 = 0.97 - 0.98
        optimizer = Lion(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            decoupled_weight_decay=True
        )
    else:
        # For most experiments we used Adam optimizer with learning rate 10−3,
        # weight decay 0, β1 = 0.9, β2 = 0.98
        optimizer = getattr(torch.optim, args.optimizer)(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)
    total_epochs = int(args.budget) // steps_per_epoch
    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data.T, True), (valid_data.T, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=0)
            for input_batch in dl:
                input_batch = input_batch.to(device)

                with torch.set_grad_enabled(is_train):
                    # logits usually [batch_size, seq_len-1, vocab_size]
                    inputs = input_batch
                    labels = torch.full_like(inputs, -100)
                    labels[:, -1] = inputs[:, -1]

                    out = model(inputs, labels=labels)
                    logits = out.logits
                    loss = out.loss
                    total_loss += loss.item() * input_batch.shape[0]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    ## CLip layers to max_norm magnitude ##
                    if args.max_norm > 0:
                        with torch.no_grad():
                            clip_weight_norms(model, args.max_norm)
                    #######################################

                acc = (logits[:, -2, :].argmax(-1) == input_batch[:, -1]).float().mean()
                total_acc += acc.item() * input_batch.shape[0]
            if is_train:
                train_acc.append(total_acc / train_data.shape[1])
                train_loss.append(total_loss / train_data.shape[1])
            else:
                val_acc.append(total_acc / valid_data.shape[1])
                val_loss.append(total_loss / valid_data.shape[1])

        if (args.plot_progress and (e + 1) % 100 == 0) or ((e + 1) >= total_epochs):
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(", ".join([
                f"model: {model_name}",
                f"optim: {args.optimizer}",
                f"lr: {args.lr:.0e}",
                f"init_pattern: {args.init_pattern if args.init_norm else 'None'}",
                f"init_norm: {args.init_norm or 'None'}",
                f"max_norm: {args.max_norm or 'None'}",
                f"task: {args.task}",
            ]))
            fig.text(0.5, 0.89, ", ".join([
                f"params: {model_num_params:,}",
                f"β1: {args.beta1:.2f}",
                f"β2: {args.beta2:.2f}",
                f"wd: {args.weight_decay:.2f}",
                f"seed: {torch.initial_seed()}",
                f"batch: {args.batch_size}",
            ]), ha='center', fontsize=10)

            ax1.plot(steps, train_loss, label="train")
            ax1.plot(steps, val_loss, label="val")
            ax1.legend()
            ax1.set_title("Loss")
            ax1.set_xlabel("Optimization Steps")
            ax1.set_ylabel("Loss")
            ax1.set_xscale("log", base=10)

            ax2.plot(steps, train_acc, label="train")
            ax2.plot(steps, val_acc, label="val")
            ax2.legend()
            ax2.set_title("Accuracy")
            ax2.set_xlabel("Optimization Steps")
            ax2.set_ylabel("Accuracy")
            ax2.set_xscale("log", base=10)

            plt.tight_layout()
            plt.show()
            fig.savefig(
                f"{args.model}_optim_{args.optimizer}-lr_{args.lr:.1e}-clip_{args.max_norm:.2f}-task_{args.task}.png")
            plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="goblin", choices=["goblin", "llama"])

    parser.add_argument("--task", type=str, default="mul-p97",
                        choices=["add-p97", "sub-p97", "mul-p97", "div-p97", "all-mod", "S5"])
    parser.add_argument("--budget", type=int, default=2e3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # Model configuration
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # Optimizer controls
    parser.add_argument("--optimizer", default="Lion")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.97)

    # Seed options
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Clip controls
    parser.add_argument("--init_pattern", type=str, default="all", choices=["all", "edge"])
    parser.add_argument("--init_norm", type=float, default=1.0)  # 0 = disable
    parser.add_argument("--max_norm", type=float, default=1.0)  # 0 = disable

    parser.add_argument("--plot_progress", action="store_true")

    args = parser.parse_args()
    main(args)
