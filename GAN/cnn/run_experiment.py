import sys

sys.path.append(".")


from GAN.cnn.train import train_model
import neptune
import argparse
from utils.keys import NEPTUNE_API

neptune.init(
    project_qualified_name="aparkhi/NoveltyGAN",
    api_token=NEPTUNE_API,
)


train_samples = [
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
    neptune.create_experiment(tags=["RDV-CNN", dataset])
    neptune.log_text("INFO", f"Epochs = {args.epochs}")
    for size in train_samples:
        print(f"For labeled size = {size}")
        func_args = argparse.Namespace(
            **{
                "train_samples": size,
                "epochs": args.epochs,
                "encoder": args.encoder,
                "webis": True if args.webis else False,
                "dlnd": True if args.dlnd else False,
                "seed": args.seed,
            }
        )
        metrics = train_model(func_args)[0]
        acc = metrics["test_acc"]
        f1 = metrics["test_f1"]
        neptune.log_metric("Metrics", acc)
        neptune.log_text("INFO", f"Labeled Size = {size}, Acc = {acc}, F1 = {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN-HAN Training")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--seed", type=int, default=42, help="Epochs")
    parser.add_argument("--webis", action="store_true", help="Webis dataset")
    parser.add_argument("--dlnd", action="store_true", help="DLND dataset")
    parser.add_argument("--encoder", type=str, help="Encoder Type")
    args = parser.parse_args()
    main(args)
