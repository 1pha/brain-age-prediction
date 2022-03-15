import argparse
from datetime import datetime

import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import MyDataset
from src.evaluate import eval, loss_plot, result_plot
from src.model.paper_ieee.levakov import Levakov
from src.model.vanilla import Vanilla3d
from src.run import run
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Age Prediction of sMRI Data")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="vanilla",
    help="Which model to use, default=vanilla",
)
parser.add_argument(
    "--task_type",
    "-t",
    type=str,
    default="binary",
    help="Task type between Classification of 2/9 classes, Regression",
)
parser.add_argument(
    "--learning_rate",
    "-lr",
    type=float,
    default=1e-4,
    help="Learning Rate, default=1e-4",
)
parser.add_argument(
    "--optimizer",
    "-o",
    type=str,
    default="adam",
    help="Type of Optimizer, default=adam",
)
parser.add_argument(
    "--scheduler", "-sc", type=str, default=None, help="Type of Scheduler, default=None"
)
parser.add_argument(
    "--loss_function",
    "-l",
    type=str,
    default="bce",
    help="Define Loss function, default=bce",
)
parser.add_argument(
    "--epochs", "-e", type=int, default=20, help="Number of epochs, default=20"
)
parser.add_argument(
    "--batch_size", "-b", type=int, default=8, help="Size of the minibatch, default=8"
)
parser.add_argument(
    "--resize",
    "-r",
    type=int,
    default=64,
    help="Size of resizing, default=64. Original raw data is 256-cube",
)
parser.add_argument("--save", "-s", action="store_true", default=True)
args = parser.parse_args()

model_list = {
    "vanilla": Vanilla3d,
    "levakov": Levakov,
}

optimizer_list = {"adam": optim.Adam}

loss_list = {"mse": nn.MSELoss, "bce": nn.BCELoss}

scheduler_list = {"cosine": optim.lr_scheduler.CosineAnnealingLR}

if __name__ == "__main__":

    # 00. Load Data
    print("# 00. Loading Data")
    train_dset = MyDataset(args.task_type)
    test_dset = MyDataset(args.task_type, test=True)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size)

    # 01. Define Model
    print("# 01. Define Model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = model_list[args.model](args.task_type).to(device)

    # 02. Setups
    print("# 02. Setups - Optim, Loss, Scheduler")
    optimizer = optimizer_list[args.optimizer](
        model.parameters(), lr=args.learning_rate
    )
    scheduler = (
        scheduler_list[args.scheduler](optimizer, len(train_loader), eta_min=0)
        if args.scheduler
        else None
    )
    loss_fn = loss_list[args.loss_function]()
    EPOCHS = range(args.epochs)

    # 03. Run
    print("# 03. Run Epochs")
    summary = (
        SummaryWriter(f'./tensorboard/{datetime.now().strftime("%Y-%m-%d_%H%M")}')
        if args.save
        else None
    )
    fname = f'{datetime.now().strftime("%Y-%m-%d_%H%M")}' if args.save else None
    model, losses, trns, preds = run(
        model=model,
        epochs=EPOCHS,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        resize=args.resize,
        summary=summary,
        scheduler=scheduler,
        verbose=True,
    )

    # 04. Evaluate
    # print("# 04. Loss Plot")
    # 04-1. Loss Plot
    # loss_plot(*losses, EPOCHS, args.loss_function)

    # 04-2. Result Plot - NOT USED(Too much memory allocation and time loss)

    sns_plot = sns.heatmap(confusion_matrix(*trns), annot=True)
    if fname:
        figure = sns_plot.get_figure()
        figure.savefig(f"./result/{fname}_{title}_binary.png", dpi=400)

    sns_plot = sns.heatmap(confusion_matrix(*preds), annot=True)
    if fname:
        figure = sns_plot.get_figure()
        figure.savefig(f"./result/{fname}_{title}_binary.png", dpi=400)

    # train_true, train_pred = eval(model=model, loader=train_loader, device=device)
    # result_plot(task_type=args.task_type, trues=train_true, preds=train_pred, title='Train', fname=fname)

    # test_true, test_pred = eval(model=model, loader=test_loader, device=device)
    # result_plot(task_type=args.task_type, trues=test_true, preds=test_pred, title='Test', fname=fname)
