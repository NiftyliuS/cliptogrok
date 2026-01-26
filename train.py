import math
from argparse import ArgumentParser
from lion_pytorch import Lion
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from norms import project_to_sphere, clip_weight_norms


class Block(nn.Module):
    """
    Causal transformer block
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class Decoder(nn.Module):
    """
    Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def division_mod_p_data(p, eq_token, op_token):
    """
    x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x * y % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])


def main(args):
    if args.random_seed:
        args.seed = torch.seed()
    else:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token = args.p + 1

    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
    ).to(device)

    #### Norm layers to equal magnitude ####
    if args.init_norm > 0:
        with torch.no_grad():
            project_to_sphere(model, args.init_norm)
    ########################################

    # "We train on the binary operation of division mod 'args.p' with 'args.train_ratio' of the data
    # in the training set."
    data = division_mod_p_data(args.p, eq_token, op_token)

    # train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2) # original code

    ##### Configurable train to validation data ratio #####
    train_size = int(data.shape[1] * args.train_ratio)
    perm = torch.randperm(data.shape[1])
    train_idx, valid_idx = perm[:train_size], perm[train_size:]
    #######################################################

    train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    ### Lion optimizer inclusion ###
    if args.optimizer == 'Lion':
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

    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    ## CLip layers to max_norm magnitude ##
                    if args.max_norm > 0:
                        with torch.no_grad():
                            clip_weight_norms(model, args.max_norm)
                    #######################################

                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])

        if (e + 1) % 100 == 0:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(", ".join([
                f"optim: {args.optimizer}",
                f"lr: {args.lr:.0e}",
                f"init_norm: {args.init_norm or 'None'}",
                f"max_norm: {args.max_norm or 'None'}",
            ]))
            fig.text(0.5, 0.89, ", ".join([
                f"β1: {args.beta1:.2f}",
                f"β2: {args.beta2:.2f}",
                f"seed: {torch.initial_seed()}"
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
            plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # Optimizer controls
    parser.add_argument("--optimizer", default="Lion")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.97)

    # Clip to Grok specific arguments
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_norm", type=float, default=2.0)  # 0 = disable
    parser.add_argument("--max_norm", type=float, default=2.0)  # 0 = disable

    args = parser.parse_args()
    main(args)
