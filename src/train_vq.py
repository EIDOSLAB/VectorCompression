
import random
import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from utils import AverageMeter, save_checkpoint, CustomDataParallel, RandomTensorDataset
from args_vq import parse_args
from models.vq_autoencoder import AutoEncoderVQ
import torch.nn as nn
import torch.nn.functional as F


class RateDistortionVQLoss(nn.Module):

    def __init__(self, alpha=2.5):
        super().__init__()
        self.alpha = alpha


    def forward(self, pred, gold, vq_loss):
        out = {}

        out['vq_loss'] = vq_loss
        out['mse_loss'] = F.mse_loss(pred, gold, reduction='mean') 

        out['loss'] = out['mse_loss'] + self.alpha*out['vq_loss']
        return out



def train_one_epoch( model: AutoEncoderVQ, criterion: RateDistortionVQLoss, train_dataloader, optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()

        x_hat, commit_loss = model(d)


        out_criterion = criterion(x_hat, d, commit_loss)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tVQ loss: {out_criterion["vq_loss"].item():.2f} |\n'
            )

def test_epoch(epoch, test_dataloader, model:AutoEncoderVQ, criterion:RateDistortionVQLoss):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    vq_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            x_hat, commit_loss = model(d)
            out_criterion = criterion(x_hat, d, commit_loss)

            vq_loss.update(out_criterion["vq_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tVQ loss: {vq_loss.avg:.2f} |\n"
    )
    # print(epoch)
    # print(loss.avg)
    # print(mse_loss.avg)
    # print(vq_loss.avg)


    return loss.avg


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    train_dataset = RandomTensorDataset(num_samples=1000, N = 192, C = 60)
    test_dataset = RandomTensorDataset(num_samples=100, N = 192, C = 60)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = AutoEncoderVQ(input_dim=60, hierarchical_structure=False, codebook_size=args.codebook_size)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)


    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionVQLoss(alpha=args.vq_alpha)


    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            os.makedirs(args.save_dir, exist_ok=True)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                save_dir = args.save_dir
            )




if __name__ == '__main__':
    main()