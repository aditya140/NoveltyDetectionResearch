# %%
import pandas as pd


def create_error_report(file_name):

    dlnd_data = pd.read_csv("dlnd_data.csv")
    analysis_file = pd.read_csv(file_name)

    analysis_file["pred_cls"] = (1 - analysis_file["pred"]).apply(lambda x: round(x))
    error_ids = analysis_file[analysis_file["pred_cls"] != analysis_file["true"]]

    error_report = dlnd_data[dlnd_data["id"].isin(error_ids.id.values)]
    error_report.to_csv(file_name.split(".")[0] + "_report." + file_name.split(".")[1])


# %%
create_error_report("han_analysis.csv")
# %%
create_error_report("han_cnn_analysis.csv")
# %%
create_error_report("diin_analysis.csv")
# %%
create_error_report("dan_analysis.csv")
# %%
create_error_report("cnn_analysis.csv")


# %%
import pandas as pd

han_cnn_rep = pd.read_csv("han_cnn_analysis_report.csv")
han_rep = pd.read_csv("han_analysis_report.csv")
dan_rep = pd.read_csv("dan_analysis_report.csv")
cnn_rep = pd.read_csv("cnn_analysis_report.csv")
diin_rep = pd.read_csv("diin_analysis_report.csv")

# %%

han_analysis = pd.read_csv("han_analysis.csv")
han_cnn_analysis = pd.read_csv("han_cnn_analysis.csv")
dan_analysis = pd.read_csv("dan_analysis.csv")
cnn_analysis = pd.read_csv("cnn_analysis.csv")
diin_analysis = pd.read_csv("diin_analysis.csv")

# %%
def intersection_ids(a, b):
    return set(a.id).intersection(b.id)


# %%
print("HAN_CNN errors - ", len(han_cnn_rep.id))
print("DAN errors - ", len(dan_rep.id))
print("HAN errors - ", len(han_rep.id))
print("CNN errors - ", len(cnn_rep.id))
print("DIIN errors - ", len(diin_rep.id))
# %%

print("HAN - CNN common errors - ", len(intersection_ids(han_rep, cnn_rep)))
print("HAN - DAN common errors - ", len(intersection_ids(han_rep, dan_rep)))
print("HAN - HAN_CNN common errors - ", len(intersection_ids(han_rep, han_cnn_rep)))
print("HAN - DIIN common errors - ", len(intersection_ids(han_rep, diin_rep)))
print("")

print("DAN - CNN common errors - ", len(intersection_ids(dan_rep, cnn_rep)))
print("DAN - HAN_CNN common errors - ", len(intersection_ids(dan_rep, han_cnn_rep)))
print("DAN - DIIN common errors - ", len(intersection_ids(dan_rep, diin_rep)))
print("")

print("HAN_CNN - CNN common errors - ", len(intersection_ids(han_cnn_rep, cnn_rep)))
print("HAN_CNN - CNN common errors - ", len(intersection_ids(han_cnn_rep, diin_rep)))
print("")

print("CNN - DIIN common errors - ", len(intersection_ids(diin_rep, cnn_rep)))
print("")
# %%

import nltk
import matplotlib.pyplot as plt


def report_analysis(rep):
    target_num_words = rep["target_text"].apply(lambda x: len(nltk.word_tokenize(x)))
    target_num_sent = rep["target_text"].apply(lambda x: len(nltk.sent_tokenize(x)))
    source_num_words = rep["source"].apply(lambda x: len(nltk.word_tokenize(x)))
    source_num_sent = rep["source"].apply(lambda x: len(nltk.sent_tokenize(x)))
    analysis = pd.DataFrame(
        {
            "target_num_words": target_num_words,
            "target_num_sent": target_num_sent,
            "source_num_words": source_num_words,
            "source_num_sent": source_num_sent,
        }
    )

    analysis.hist()
    return analysis.describe(percentiles=[])


# %%
# HAN REPORT
report_analysis(han_rep)
# %%
# HAN_CNN Report
report_analysis(han_cnn_rep)
# %%
# DAN Report
report_analysis(dan_rep)
# %%
# CNN Report
report_analysis(cnn_rep)
# %%
# DIIN Report
report_analysis(diin_rep)
# %%
# HAN - HAN_CNN common errors
report_analysis(han_rep.merge(han_cnn_rep))
# HAN - DAN common errors
report_analysis(han_rep.merge(dan_rep))
# %%
# HAN - CNN common errors
report_analysis(han_rep.merge(cnn_rep))
# %%
# DAN - HAN_CNN common errors
report_analysis(dan_rep.merge(han_cnn_rep))
# %%
# DAN - CNN common errors
report_analysis(dan_rep.merge(cnn_rep))
# %%
# CNN - HAN_CNN common errors
report_analysis(cnn_rep.merge(han_cnn_rep))

# %%

dlnd_data = pd.read_csv("dlnd_data.csv")

with open("./analysis/han_cnn.txt", "r") as f:
    ids = [int(i) for i in f.read().split(",")]

# %%

len(set(ids))


# %%
han_cnn_all_errors = dlnd_data[dlnd_data["id"].isin(set(ids))]
# %%

report_analysis(han_cnn_all_errors)
# %%


han_cnn_all_errors.to_csv("./han_cnn_10_fold_all_errors.csv")
# %%
