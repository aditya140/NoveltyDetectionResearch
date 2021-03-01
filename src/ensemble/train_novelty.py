import sys

sys.path.append(".")

from torchensemble.utils.logging import set_logger
from src.ensemble.ensemble import *

from src.defaults import *
from src.datasets.novelty import *
from src.model.nli_models import *
from src.model.novelty_models import *


from hyperdash import Experiment
import neptune


class Train:
    def __init__(
        self,
        args,
        dataset_conf,
        model_conf,
        hparams,
        model_type,
        sentence_field,
    ):
        self.args = args

        neptune.init(
            project_qualified_name=NOVELTY_ENSEMBLE_NEPTUNE_PROJECT,
            api_token=NEPTUNE_API,
        )
        self.exp = neptune.create_experiment()
        self.exp_id = self.exp.id

        neptune.log_text("Dataset Conf", str(dataset_conf))
        neptune.log_text("Model Conf", str(model_conf))
        neptune.log_text("Hparams", str(hparams))

        self.hd_exp = Experiment(
            NOVELTY_ENSEMBLE_NEPTUNE_PROJECT, api_key_getter=get_hyperdash_api
        )
        self.hd_exp.param("Dataset Conf", str(dataset_conf))
        self.hd_exp.param("Model Conf", str(model_conf))
        self.hd_exp.param("Hparams", str(hparams))

        set_logger("ensemble_test")
        self.dataset = novelty_dataset(dataset_conf, sentence_field=sentence_field)

        nli_model_data = load_encoder_data(args.load_nli)
        encoder = self.load_encoder(nli_model_data).encoder
        model_conf["encoder_dim"] = nli_model_data["options"]["hidden_size"]

        if model_type == "dan":
            model = DAN
        if model_type == "adin":
            model = ADIN
        if model_type == "han":
            model = HAN
        if model_type == "rdv_cnn":
            model = RDV_CNN
        if model_type == "diin":
            model = DIIN
        if model_type == "mwan":
            model = MwAN
        if model_type == "struc":
            model = StrucSelfAttn

        self.model = VotingClassifier_novelty(
            estimator=model,
            n_estimators=5,
            cuda=True,
            estimator_args={"conf": model_conf, "encoder": encoder},
        )

        if hparams["optimizer"]["optim"] == "adam":
            self.model.set_optimizer("Adam", lr=hparams["optimizer"]["lr"])

        if hparams["optimizer"]["optim"] == "adamw":
            self.model.set_optimizer("AdamW", lr=hparams["optimizer"]["lr"])

        if hparams["optimizer"]["optim"] == "sgd":
            self.model.set_optimizer("SGD", lr=hparams["optimizer"]["lr"])

        if hparams["optimizer"]["scheduler"] == "step":
            self.model.set_scheduler("StepLR", step_size=5, gamma=0.5)

        elif hparams["optimizer"]["scheduler"] == "cosine":
            self.model.set_scheduler("CosineAnnealingLR", T_max=10)

    def execute(self):
        train_loader = self.dataset.train_iter
        val_loader = self.dataset.val_iter
        test_loader = self.dataset.test_iter

        self.model.fit(train_loader, epochs=self.args.epochs, test_loader=val_loader)
        test_acc = self.model.predict(test_loader)
        print("Test Acc:", test_acc)
        self.hd_exp.log(f"Test Acc: {test_acc}")
        self.exp.log(f"Test Acc: {test_acc}")

    @staticmethod
    def load_encoder(enc_data):
        if enc_data["options"].get("attention_layer_param", 0) == 0:
            enc_data["options"]["use_glove"] = False
            model = bilstm_snli(enc_data["options"])
        elif enc_data["options"].get("r", 0) == 0:
            enc_data["options"]["use_glove"] = False
            model = attn_bilstm_snli(enc_data["options"])
        else:
            enc_data["options"]["use_glove"] = False
            model = struc_attn_snli(enc_data["options"])

        model.load_state_dict(enc_data["model_dict"])
        return model


if __name__ == "__main__":
    args = parse_novelty_conf()
    dataset_conf, optim_conf, model_type, model_conf, sentence_field = get_novelty_conf(
        args
    )
    trainer = Train(
        args, dataset_conf, model_conf, optim_conf, model_type, sentence_field
    )
    trainer.execute()
