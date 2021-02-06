import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.profiler import AdvancedProfiler

sys.path.append(".")

from src.defaults import *
from src.datasets.nli import *
from src.model.nli_models import *
from src.utils.nli_utils import *


if __name__ == "__main__":
    seed_torch()
    args = parse_nli_conf()
    dataset_conf, optim_conf, model_type, model_conf = get_conf(args)

    if dataset_conf["dataset"] == "snli":
        datamodule = snli_module(dataset_conf)
    elif dataset_conf["dataset"] == "mnli":
        datamodule = mnli_module(dataset_conf)
    datamodule.prepare_data()

    model_conf["vocab_size"] = datamodule.vocab_size()
    model_conf["padding_idx"] = datamodule.padding_idx()

    if model_type == "attention":
        model = attn_bilstm_snli
    elif model_type == "bilstm":
        model = bilstm_snli

    pl_model = NLI_model(model, model_conf, optim_conf)

    EPOCHS = optim_conf["epochs"]

    tensorboard_logger = TensorBoardLogger("lightning_logs")
    lr_logger = LearningRateLogger(logging_interval="step")

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API,
        project_name="aparkhi/NLI",
        experiment_name="Training",
        tags=[dataset_conf["dataset"], model_type],
    )
    expt_id = neptune_logger.experiment.id
    neptune_logger.experiment.log_metric("epochs", EPOCHS)
    neptune_logger.experiment.log_text("optim_conf", optim_conf.__str__())
    neptune_logger.experiment.log_text("model_conf", model_conf.__str__())
    neptune_logger.experiment.log_text("dataset_conf", dataset_conf.__str__())

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        progress_bar_refresh_rate=10,
        profiler=False,
        auto_lr_find=False,
        callbacks=[lr_logger, SwitchOptim()],
        logger=[tensorboard_logger, neptune_logger],
        row_log_interval=2,
        gradient_clip_val=0.5,
    )

    trainer.fit(pl_model, datamodule)
    trainer.test(pl_model, datamodule=datamodule)
