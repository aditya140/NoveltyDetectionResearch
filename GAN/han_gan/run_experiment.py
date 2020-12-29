import sys

sys.path.append(".")


import neptune
import os
from GAN.han_gan.train import train_model
import argparse
from keys import NEPTUNE_API

neptune.init(
    project_qualified_name="aditya140/GANcompare",
    api_token=NEPTUNE_API,
)


labeled_sizes = [
    2,
    4,
    6,
    8,
    10,
    16,
    32,
    46,
    64,
    78,
    100,
    120,
    160,
    200,
    260,
    300,
    350,
]


def main(args):
    epochs = args.epochs
    dataset = "Webis" if args.webis else "DLND"
    neptune.create_experiment(tags=["HAN-GAN", dataset])
    neptune.log_text("INFO", f"Epochs = {args.epochs}")
    for size in labeled_sizes:
        print(f"For labeled size = {size}")
        func_args = argparse.Namespace(
            **{
                "labeled_size": size,
                "epochs": args.epochs,
                "log": False,
                "encoder": args.encoder,
                "webis": True if args.webis else False,
                "dlnd": True if args.dlnd else False,
            }
        )
        acc = train_model(func_args)
        neptune.log_metric("Test Accuracy", acc)
        neptune.log_text("INFO", f"Labeled Size = {size}, Acc = {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN-HAN Training")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train Size")
    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    args = parser.parse_args()
    main(args)
