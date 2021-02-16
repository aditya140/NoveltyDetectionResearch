import sys

sys.path.append(".")

from torchensemble.utils.logging import set_logger
from src.ensemble.ensemble import *

from src.defaults import *
from src.datasets.novelty import *
from src.model.nli_models import *
from src.model.novelty_models import *


# download_models_from_neptune("NLI-81")
field = load_field("NLI-87")
# field = None


dataset_conf = {
    "dataset": "dlnd",
    "max_num_sent": 60,
    "sent_tokenizer": "spacy",
    "batch_size": 16,
    "device": "cuda",
}
# dataset_conf = {'dataset': 'dlnd', 'max_num_sent': 50,"sent_tokenizer":"spacy", "tokenizer":'spacy',"max_len":50,"batch_size":32,"device":"cuda"}
model_conf = {
    "results_dir": "results",
    "device": "cuda",
    "dropout": 0.3,
    "dataset": "dlnd",
    "hidden_size": 400,
    "use_glove": False,
    "max_num_sent": 60,
}


data = dlnd(dataset_conf, sentence_field=field)


def load_encoder(enc_data):
    if enc_data["options"].get("attention_layer_param", 0) == 0:
        model = bilstm_snli(enc_data["options"])
    elif enc_data["options"].get("r", 0) == 0:
        model = attn_bilstm_snli(enc_data["options"])
    else:
        model = struc_attn_snli(enc_data["options"])
    return model


def load_model_data(_id):
    model_path = os.path.join("./results/", _id, "model.pt")
    model_data = torch.load(model_path)
    return model_data


model_data = load_model_data("NLI-81")

model_conf["encoder_dim"] = model_data["options"]["hidden_size"]
model_data["options"]["use_glove"] = False

model = load_encoder(model_data)


ens = VotingClassifier_novelty(
    estimator=DAN,
    n_estimators=3,
    cuda=True,
    estimator_args={"conf": model_conf, "encoder": model.encoder},
)


ens.set_optimizer("AdamW", lr=0.001)


train_loader = data.train_iter
val_loader = data.val_iter
set_logger("ensemble_test")
ens.fit(train_loader, epochs=10, test_loader=val_loader)


print("Test Acc:", ens.predict(data.test_iter))
