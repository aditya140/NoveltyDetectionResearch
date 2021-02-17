import abc
import torch
from src.defaults import *
import neptune


class Trainer(abc.ABC):
    def __init__(self, log_neptune=True, **kwargs):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.log_neptune = log_neptune
        if log_neptune:
            neptune.init(
                project_qualified_name=NOVELTY_NEPTUNE_PROJECT,
                api_token=NEPTUNE_API,
            )
            neptune.log_text("Dataset Conf", str(dataset_conf))
            neptune.log_text("Model Conf", str(model_conf))
            neptune.log_text("Hparams", str(hparams))

        self.args = args
        self.device = torch.device("cuda" if cuda else "cpu")

        self.logger = get_logger(self.args, "train", self.exp_id)

        self.logger.info("Dataset Conf: {}".format(dataset_conf))
        self.logger.info("Model Conf: {}".format(model_conf))
        self.logger.info("Hparams Conf: {}".format(hparams))

        self.model_conf = model_conf
        self.dataset_conf = dataset_conf

    @abc.abstractmethod
    def load_dataset(self, dataset_conf, **kwargs):
        """
        Implementation on the process of loading dataset.
        """

    @abc.abstractmethod
    def laod_model(self, model_conf, **kwargs):
        """
        Implementation on the process of loading dataset.
        """

    @abc.abstractmethod
    def set_optimizers(self, hparams, **kwargs):
        """
        Implementation on the process of loading optimizers.
        """

    @abc.abstractmethod
    def set_schedulers(self, hparams, **kwargs):
        """
        Implementation on the process of loading schedulers.
        """

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print("-" * 99)
        print(" [*] Training finished!")
        print(" [*] Please find the training log in results_dir")

    def train(self, train_iterator, **kwargs):
        self.model.train()
        train_iterator.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        if kwargs.get("batch_attr",None)==None:
            batch_attr = None
        else:
            batch_attr = kwargs["batch_attr"]

        for batch_idx, batch in enumerate(train_iterator):
            self.opt.zero_grad()
            answer = self.model(batch.source, batch.target)
            loss = self.criterion(answer, batch.label)
            n_correct += (
                (
                    torch.max(self.softmax(answer), 1)[1].view(batch.label.size())
                    == batch.label
                )
                .sum()
                .item()
            )
            n_total += batch.batch_size
            n_loss += loss.item()
            if self.log_neptune:
                neptune.log_metric("Train Loss", loss.item())
            loss.backward()
            self.opt.step()
        train_loss = n_loss / n_total
        train_acc = 100.0 * n_correct / n_total
        if self.log_neptune:
            neptune.log_metric("Train Avg Loss", train_loss)
            neptune.log_metric("Train accuracy", train_acc)
        return train_loss, train_acc
