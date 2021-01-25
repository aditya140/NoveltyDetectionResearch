# %%
from utils.load_models import *

# %%
model, lang, model_conf = load_han_novelty("NOV-189")
# %%

from novelty.cnn.aggregator import *
from novelty.cnn.cnn_model import *
from snli.bilstm.bilstm import *
import pickle
import joblib
import torch
import shutil
import os
from tqdm.auto import tqdm
import math
import nltk


def batch(iterable, n=32):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def to_novelty_tensor(s, Lang, model_conf):
    if isinstance(s, list):
        s = ". ".join(s)
    pad_arr = [0] * Lang.max_len
    t = Lang.encode_batch(nltk.sent_tokenize(s))
    if t.shape[0] < model_conf.num_sent:
        opt = np.append(t, [pad_arr] * (model_conf.num_sent - t.shape[0]), axis=0)
    else:
        opt = t[
            : model_conf.num_sent,
        ]
    return opt


def predict_novelty(source, target, model, cuda=True, max_batch_size=32):
    opt = []
    for inp_batch in tqdm(
        batch(list(zip(source, target)), n=32), total=math.ceil(len(source) / 32)
    ):
        src, trg = map(list, zip(*inp_batch))
        src_vec = []
        for i in src:
            src_vec.append(
                torch.tensor(to_novelty_tensor(i, model["lang"], model["model_conf"]))
            )
        trg_vec = []
        for j in trg:
            trg_vec.append(
                torch.tensor(to_novelty_tensor(j, model["lang"], model["model_conf"]))
            )
        src_vec = torch.stack(src_vec)
        trg_vec = torch.stack(trg_vec)

        if cuda:
            src_vec = src_vec.cuda()
            trg_vec = trg_vec.cuda()
        opt += (
            torch.nn.functional.softmax(model["model"](trg_vec, src_vec))
            .cpu()
            .detach()
            .tolist()
        )
    return opt


def predict_novelty(source, target, model, cuda=True, max_batch_size=32):
    opt = []
    for inp_batch in tqdm(
        batch(list(zip(source, target)), n=32), total=math.ceil(len(source) / 32)
    ):
        src, trg = map(list, zip(*inp_batch))
        src_vec = []
        for i in src:
            src_vec.append(
                torch.tensor(to_novelty_tensor(i, model["lang"], model["model_conf"]))
            )
        trg_vec = []
        for j in trg:
            trg_vec.append(
                torch.tensor(to_novelty_tensor(j, model["lang"], model["model_conf"]))
            )
        src_vec = torch.stack(src_vec)
        trg_vec = torch.stack(trg_vec)

        if cuda:
            src_vec = src_vec.cuda()
            trg_vec = trg_vec.cuda()
        opt += (
            torch.nn.functional.softmax(model["model"](trg_vec, src_vec))
            .cpu()
            .detach()
            .tolist()
        )
    return opt


# %%
model_conf.num_sent = 100
model_dict = {"model": model.cuda(), "lang": lang, "model_conf": model_conf}
# %%
predict_novelty(["Hello how are you"], ["what is your"], model_dict)
# %%


def map_attention(source, target, model, cuda=True, max_batch_size=32):
    src_vec = []

    src_vec.append(
        torch.tensor(to_novelty_tensor(source, model["lang"], model["model_conf"]))
    )
    trg_vec = []

    trg_vec.append(
        torch.tensor(to_novelty_tensor(target, model["lang"], model["model_conf"]))
    )

    src_vec = torch.stack(src_vec)
    trg_vec = torch.stack(trg_vec)

    if cuda:
        src_vec = src_vec.cuda()
        trg_vec = trg_vec.cuda()

    model_opt, src_attn, trg_attn = model["model"].forward_with_attn(trg_vec, src_vec)
    opt = torch.nn.functional.softmax(model_opt).cpu().detach().tolist()

    return opt, src_attn, trg_attn


# %%
map_attention("Hello how are you", "what is your", model_dict)
# %%
import pandas as pd

data = pd.read_csv("dlnd_data.csv")

# %%
row = data.iloc[1]
target_text = row.target_text
source_text = row.source
DLA = row.DLA

# %%
opt, src_att, trg_att = map_attention(source_text, target_text, model_dict)

# %%
len(nltk.sent_tokenize(source_text))
# %%
opt
# %%
DLA
# %%
row = data.iloc[5347]
target_text = row.target_text
source_text = row.source
DLA = row.DLA
# %%
opt, src_att, trg_att = map_attention(source_text, target_text, model_dict)
# %%
opt
# %%
src_att
# %%
trg_att
# %%
import pandas as pd
import numpy as np
import html
import random
from IPython.core.display import display, HTML

# %%
# Prevent special characters like & and < to cause the browser to display something other than what you intended.
def html_escape(text):
    return html.escape(text)


# %%
# Taken from :http://52.51.209.151/data-analysis-resources/an-analysis-of-the-impact-of-eu-membership-on-the-economic-development-of-ireland/ http://52.51.209.151/data-analysis-resources/an-analysis-of-the-impact-of-eu-membership-on-the-economic-development-of-ireland/


text = "Ireland became a member of the European Union (EU) and joined the single market on 1st January 1973. Before accession to the bloc, Ireland had decades of an underachieving economy which was heavily dependent on the UK. Since then it has transformed into a prosperous and confident country which is a major influence in the global politics. The economy has transformed from agricultural dependent to one driven by the tech industry and global exports. The membership has also affected every part of Irish society from the way the citizens work, travel or even shop [1].However, the recent political turmoil of Brexit and 2008 recession crisis has left certain citizens to wonder the importance of the membership. Sometimes, there is a doubt in EUs ability to provide a good living standard. Economic development requires economic growth to reflect economic as well as social growth. Since joining the EU, Ireland has developed economically and changed socially. The statistical and mathematical analysis of the data from World Bank shows that the economy has transformed from agriculture dependent to dependent on manufacturing merchandise, goods and services industry. There has been growth in the population due to reduced death rate and increased life expectancy. The birth rate has also decreased. The population is young and highly educated which helps the multinational companies making decision to base their operation in Ireland. The participation of females in the job market has increased over the years but females are also most likely to be employed in part-time jobs"
# %%
# Remove duplicate words from text
seen = set()
result = []
for item in text.split():
    if item not in seen:
        seen.add(item)
        result.append(item)
# %%
# Create random sample weights for each unique word
weights = []
for i in range(len(result)):
    weights.append(random.random())


df_coeff = pd.DataFrame({"word": result, "num_code": weights})

# %%


# Select the code value to generate different weights
word_to_coeff_mapping = {}
for row in df_coeff.iterrows():
    row = row[1]
    word_to_coeff_mapping[row[1]] = row[0]
# %%
max_alpha = 0.8
highlighted_text = []
for word in text.split():
    weight = word_to_coeff_mapping.get(word)

    if weight is not None:
        highlighted_text.append(
            '<span style="background-color:rgba(135,206,250,'
            + str(weight / max_alpha)
            + ');">'
            + html_escape(word)
            + "</span>"
        )
    else:
        highlighted_text.append(word)
highlighted_text = " ".join(highlighted_text)
# highlighted_text
# %%
display(HTML(highlighted_text))

# %%
