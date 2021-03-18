import sys

import warnings

warnings.filterwarnings("ignore")
sys.path.append(".")

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.defaults import *
from src.datasets.novelty import *
from src.model.nli_models import *
from src.model.novelty_models import *

import sklearn
from sklearn.ensemble import BaggingClassifier, VotingClassifier


from hyperdash import Experiment
import neptune


class TorchEnsemble(sklearn.base.BaseEstimator):
    def __init__(
        self,
        net_type,
        net_params,
        optim_type,
        optim_params,
        loss_fn,
        input_shape,
        accuracy_tol=0.02,
        batch_size=32,
        tol_epochs=10,
        cuda=True,
    ):
        self.net_type = net_type
        self.net_params = net_params
        self.optim_type = optim_type
        self.optim_params = optim_params
        self.loss_fn = loss_fn

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.accuracy_tol = accuracy_tol
        self.tol_epochs = tol_epochs
        self.cuda = cuda

    def fit(self, X, y):
        self.net = self.net_type(*self.net_params)
        if self.cuda:
            self.net = self.net.cuda()
        self.optim = self.optim_type(self.net.parameters(), **self.optim_params)

        uniq_classes = np.sort(np.unique(y))
        self.classes_ = uniq_classes

        X = X.reshape(-1, *self.input_shape)
        y = y.reshape(-1, *self.input_shape)

        X_tensor = torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.long))
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        last_accuracies = []
        epoch = 0
        keep_training = True
        while keep_training:
            print(f"Epoch {epoch}")
            self.net.train()
            train_samples_count = 0
            true_train_samples_count = 0
            epoch += 1
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                x, y_data = batch[0], batch[1]
                src, trg = x[:, 0, :, :], x[:, 1, :, :]
                if self.cuda:
                    src = src.cuda()
                    trg = trg.cuda()
                    y_data = y_data.cuda()

                y_pred = self.net(src, trg)
                y_pred = F.softmax(y_pred)
                loss = self.loss_fn(y_pred, y_data)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                y_pred = y_pred.argmax(dim=1, keepdim=False)
                true_classified = (y_pred == y_data).sum().item()
                true_train_samples_count += true_classified
                train_samples_count += len(src)

            train_accuracy = true_train_samples_count / train_samples_count
            last_accuracies.append(train_accuracy)

            if len(last_accuracies) > self.tol_epochs:
                last_accuracies.pop(0)

            if epoch >= self.tol_epochs:
                keep_training = False

            if len(last_accuracies) == self.tol_epochs:
                accuracy_difference = max(last_accuracies) - min(last_accuracies)
                if accuracy_difference <= self.accuracy_tol:
                    keep_training = False

    def predict_proba(self, X, y=None):
        X = X.reshape(-1, *self.input_shape)
        y = y.reshape(-1, *self.input_shape)

        X_tensor = torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.long))
        test_dataset = TensorDataset(X_tensor, y_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        self.net.eval()
        predictions = []
        for batch in test_loader:
            x, y_data = batch[0], batch[1]
            src, trg = x[:, 0, :, :], x[:, 0, :, :]
            if self.cuda:
                src = src.cuda()
                trg = trg.cuda()
                y_data = y_data.cuda()

            y_pred = self.net(src, trg)
            y_pred = F.softmax(y_pred)
            predictions.append(y_pred.detach().cpu().numpy())

        predictions = np.concatenate(predictions)
        return predictions

    def predict(self, X, y=None):
        predictions = self.predict_proba(X, y)
        predictions = predictions.argmax(axis=1)
        return predictions


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

        dataset_conf["doc_field"] = False
        self.dataset = novelty_dataset(dataset_conf, sentence_field=sentence_field)

        self.create_dataset()

        nli_model_data = load_encoder_data(args.load_nli)
        encoder = self.load_encoder(nli_model_data).encoder
        model_conf["encoder_dim"] = nli_model_data["options"]["hidden_size"]

        model_args = [model_conf, encoder]

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
        if model_type == "matt":
            model = MultiAtt
        if model_type == "ein":
            model = EIN
        if model_type == "eain":
            model = EAtIn

        base_model = TorchEnsemble(
            net_type=model,
            net_params=model_args,
            optim_type=optim.Adam,
            optim_params={"lr": 1e-3},
            loss_fn=nn.CrossEntropyLoss(),
            input_shape=(1, 28, 28),
            accuracy_tol=0.02,
            tol_epochs=10,
            cuda=True,
        )

        self.model = BaggingClassifier(base_estimator=base_model, n_estimators=3)

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

    def create_dataset(self):
        self.train, self.val, self.test = self.dataset.get_numpy_data()
        self.inp_shape = self.train[0].shape[1:]
        self.train = self.make_numpy_array(self.train)
        self.val = self.make_numpy_array(self.val)
        self.test = self.make_numpy_array(self.test)

    @staticmethod
    def make_numpy_array(data):
        dataset_size = data[0].shape
        input_shape = dataset_size[1:]
        data[0] = data[0].reshape(dataset_size[0], -1).shape
        return data

    def execute(self):
        # Train
        X, y = self.train
        self.model.fit(X, y)

        # Test
        X_test, y_test = self.test
        preds = base_model.predict(X_test)
        true_classified = (preds == y_test).sum()
        test_accuracy = true_classified / len(y_test)

        print(f"Test accuracy: {test_accuracy}")
        self.hd_exp.log(f"Test Acc: {test_accuracy}")
        self.exp.log(f"Test Acc: {test_accuracy}")


if __name__ == "__main__":
    args = parse_novelty_conf()
    dataset_conf, optim_conf, model_type, model_conf, sentence_field = get_novelty_conf(
        args
    )
    trainer = Train(
        args, dataset_conf, model_conf, optim_conf, model_type, sentence_field
    )
    trainer.execute()
