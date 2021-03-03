import sys

sys.path.append(".")
import warnings

warnings.filterwarnings("ignore")

from src.defaults import *
import pickle, json
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

dlnd_models_to_compare = ["NOV-452", "NOV-450", "NOV-446", "NOV-445"]
# apwsj_models_to_compare = ["NOV-444"]


def get_baseline_results(dataset):
    baseline_path = f"results/novelty_baseline/{dataset}_baselines_class_probs.p"
    assert os.path.exists(baseline_path)

    results = {}
    with open(baseline_path, "rb") as f:
        data = pickle.load(f)
    baselines = [i.replace("probs_", "") for i in data.keys() if ("probs_" in i)]
    class_order = data["class_order"]  # 1 is novel
    gold = data["labels"]

    results = {}
    for i in baselines:
        probs = data[f"probs_{i}"]
        preds = np.argmax(probs, 1)
        acc = accuracy_score(gold, preds)
        p, r, f, _ = precision_recall_fscore_support(gold, preds, labels=[0, 1])
        preds_data = {"prob": probs, "gold": gold, "pred": preds}
        results[i] = {
            "model_type": i,
            "accuracy": acc,
            "recall": r,
            "prec": p,
            "f1": f,
            "class_labels": {"Non-Novel": 0, "Novel": 1},
            "preds": preds_data,
        }
    return results


def get_model_resutls(models):
    for i in dlnd_models_to_compare:
        check_model(i)

    exps = project.get_experiments(dlnd_models_to_compare)
    results = {}
    for exp in exps:
        _, model_type = exp.get_tags()
        recall = exp.get_channels()["Final Recall"].y
        prec = exp.get_channels()["Final Precision"].y
        acc = exp.get_channels()["Final Accuracy"].y
        f1 = exp.get_channels()["Final F1"].y
        class_labels = exp.get_channels()["class_labels"].y
        with open(f"results/{exp.id}/probs.p", "rb") as f:
            preds = pickle.load(f)
        results[exp.id] = {
            "model_type": model_type,
            "accuracy": eval(acc) / 100,
            "recall": eval(recall),
            "prec": eval(prec),
            "f1": eval(f1),
            "class_labels": eval(class_labels),
            "preds": preds,
        }
    return results


def plot_prc_curve(results):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    p = setup_prc_plot("Non Novel Precision Recall Curve")
    for k, v in results.items():
        k = v.get("model_type", k)
        prob = v["preds"]["prob"]
        gold = v["preds"]["gold"]
        non_novel_class = v["class_labels"]["Novel"]
        p = plot_prc(p, prob, gold, cls_label=non_novel_class, label=k)

    p.savefig("plots/NonNovel_prc.jpg")
    plt.clf()
    p = setup_prc_plot("Novel Precision Recall Curve")
    for k, v in results.items():
        k = results.get("model_type", k)
        prob = v["preds"]["prob"]
        gold = v["preds"]["gold"]
        novel_class = v["class_labels"]["Non-Novel"]
        p = plot_prc(p, prob, gold, cls_label=novel_class, label=k)

    p.savefig("plots/Novel_prc.jpg")
    plt.clf()


def process_row(x):
    x["Non Novel Precision"] = x["prec"][x["class_labels"]["Non-Novel"]]
    x["Novel Precision"] = x["prec"][x["class_labels"]["Novel"]]
    x["Non Novel Recall"] = x["recall"][x["class_labels"]["Non-Novel"]]
    x["Novel Recall"] = x["prec"][x["class_labels"]["Novel"]]
    x["Non Novel F1"] = x["f1"][x["class_labels"]["Non-Novel"]]
    x["Novel F1"] = x["f1"][x["class_labels"]["Novel"]]
    x["Accuracy"] = x["accuracy"]
    return x


def get_results_dataframe(res):
    df = pd.DataFrame(res)
    df = df.transpose()
    df = df.reset_index(drop=True)
    df = df.apply(lambda x: process_row(x), axis=1)
    df = df.drop(columns=["accuracy", "recall", "prec", "f1", "class_labels", "preds"])
    df = df.sort_values(by=["Accuracy"])
    with open("plots/result_table.log", "w") as f:
        f.write(tabulate(df, headers="keys", tablefmt="psql"))

    return df


if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset == "dlnd":
        project = neptune.init(NOVELTY_NEPTUNE_PROJECT, api_token=NEPTUNE_API)
        model_results = get_model_resutls(dlnd_models_to_compare)
        baseline_results = get_baseline_results("dlnd")
        model_results.update(baseline_results)
    plot_prc_curve(model_results)
    df = get_results_dataframe(model_results)