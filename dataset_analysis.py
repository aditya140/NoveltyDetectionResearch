# %%
import pandas as pd
import json
import os
from nltk import sentence_tokenizer

# %%
with open(".data/dlnd/TAP-DLND-1.0_LREC2018_modified/dlnd.jsonl", "r") as f:
    data = [json.loads(i) for i in f.readlines()]
# %%
source_lens = [len(nltk.sent_tokenize(i["source"])) for i in data]
target_lens = [len(nltk.sent_tokenize(i["target_text"])) for i in data]
# %%
max(source_lens)
# %%
max(target_lens)

# %%
import seaborn as sns

fig = sns.displot(source_lens)
fig.set_axis_labels("Document Length", "Count")

# %%
source = [i["source"] for i in data]
target = [i["target_text"] for i in data]
# %%
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


source_lens = [(len(sent_tokenize(i))) for i in source]
target_lens = [(len(sent_tokenize(i))) for i in target]
# %%
total_lens = [source_lens[i] + target_lens[i] for i in range(len(target_lens))]

# %%
print(max(total_lens))


# %%
import seaborn as sns

sns.displot(total_lens, kind="kde")


# %%
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# DLND sentence length bert tokenizer
source = [i["source"] for i in or_data]
target = [i["target_text"] for i in or_data]

encoding = tokenizer.encode_plus(
    source[0], add_special_tokens=False, truncation=False, padding="longest"
)["input_ids"]
print(len(encoding))
# %%
source_lens = [
    len(
        tokenizer.encode_plus(
            i, add_special_tokens=False, truncation=False, padding="longest"
        )["input_ids"]
    )
    for i in source
]
target_lens = [
    len(
        tokenizer.encode_plus(
            i, add_special_tokens=False, truncation=False, padding="longest"
        )["input_ids"]
    )
    for i in target
]
# %%
total_lens = [source_lens[i] + target_lens[i] for i in range(len(target_lens))]

# %%
print(max(total_lens))


# %%
import seaborn as sns

fig = sns.displot(target_lens)
fig.set_axis_labels("Document Length", "Count")


# %%
acc = """
0	88.97058823529412
1	87.31617647058823
2	87.68382352941177
3	88.97058823529412
4	88.05147058823529
5	87.13235294117646
6	88.23529411764706
7	87.31617647058823
8	88.05147058823529
"""

acc = [float(i.split("\t")[1]) for i in acc.strip().split("\n")]
# %%
sum(acc) / len(acc)
# %%
