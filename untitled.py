# %%
from argparse import ArgumentParser
import argparse

# %%
import matplotlib.pyplot as plt
import neptune

NEPTUNE_API = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTg3MzU5NjQtMmIxZC00Njg0LTgzYzMtN2UwYjVlYzVhNDg5In0="
NLI_NEPTUNE_PROJECT = "aparkhi/NLI"
NOVELTY_NEPTUNE_PROJECT = "aparkhi/Novelty"
NOVELTY_ENSEMBLE_NEPTUNE_PROJECT = "aparkhi/NoveltyEnsemble"
DOC_NEPTUNE_PROJECT = "aparkhi/DocClassification"
VARY_LABELED = "aparkhi/VaryLabeled"

labeled_list = [
    2,
    4,
    10,
    20,
    40,
    60,
    80,
    100,
    200,
    300,
    400,
    500,
    600,
    1000,
    2000,
    3000,
]

test_acc_list = [
    51.61,
    51.61,
    51.61,
    49.95,
    51.61,
    51.61,
    51.61,
    59.34,
    60.44,
    65.32,
    60.81,
    66.24,
    71.48,
    71.02,
    78.29,
    82.52,
]


neptune.init(
    project_qualified_name=VARY_LABELED,
    api_token=NEPTUNE_API,
)

exp = neptune.create_experiment()
exp_id = exp.id
neptune.append_tag(["dlnd", "struc"])

fig = plt.figure()
plt.plot(labeled_list, test_acc_list)
plt.title("Varying Labeled Set Size")
plt.xlabel("Labeled Set Size")
plt.ylabel("Test Accuracy")
plt.legend(["struc"])
new_path = os.path.join("plots", f"vary_labeled_struc.png")
ver = 0
if not os.path.exists("plots"):
    os.makedirs("plots")
while os.path.exists(new_path):
    ver += 1
    new_path = os.path.join("plots", f"vary_labeled_struc{str(ver)}.png")

neptune.log_image("vary_labeled_size", fig, image_name="vary_labeled_size")
fig.savefig(new_path)
neptune.log_artifact(new_path)
exp.end()
# %%
