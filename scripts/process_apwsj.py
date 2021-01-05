import os
import glob
import json
from collections import defaultdict
import gzip
from os.path import join as osj





def organize_files():
    AP_path = "./dataset/trec/AP"
    AP_files = glob.glob(osj(AP_path,"*.gz"))
    for i in AP_files:
        with gzip.open(i,'r') as f:
            text = f.read()

        with open(i[:-3],'wb') as f_new:
            f_new.write(text)
        os.remove(i)



def create_json():
    """
    """
    wsj = "./dataset/trec/trec/wsj"
    ap = "./dataset/trec/trec/ap"
    ap_others = "./dataset/trec/AP"
    wsj_others = "./dataset/trec/WSJ/wsj_split"

    wsj_files = glob.glob(wsj + "/*")
    ap_files = glob.glob(ap + "/*")


    wsj_other_files = []
    ap_other_files = glob.glob(osj(ap_others,"*"))

    wsj_big = glob.glob(osj(wsj_others,"*"))
    for i in wsj_big:
        for file_path in glob.glob(osj(i,"*")):
            wsj_other_files.append(file_path)
        

    docs_json = {}
    errors = 0
    for wsj_file in wsj_files:
        with open(wsj_file, "r") as f:
            txt = f.read()
        docs = [
            i.split("<DOC>")[1]
            for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
        ]

        for doc in docs:
            try:
                id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                docs_json[id] = text
            except:
                errors += 1

    for ap_file in ap_files:
        with open(ap_file, "r", encoding="latin-1") as f:
            txt = f.read()
        docs = [
            i.split("<DOC>")[1]
            for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
        ]

        for doc in docs:
            try:
                id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                docs_json[id] = text
            except:
                errors += 1



    for wsj_file in wsj_other_files:
        with open(wsj_file, "r") as f:
            txt = f.read()
        docs = [
            i.split("<DOC>")[1]
            for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
        ]

        for doc in docs:
            try:
                id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                docs_json[id] = text
            except:
                errors += 1

    for ap_file in ap_other_files:
        with open(ap_file, "r", encoding="latin-1") as f:
            txt = f.read()
        docs = [
            i.split("<DOC>")[1]
            for i in filter(lambda x: len(x) > 10, txt.split("</DOC>"))
        ]

        for doc in docs:
            try:
                id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0]
                text = doc.split("<TEXT>")[1].split("</TEXT>")[0]
                docs_json[id] = text
            except:
                errors += 1


    print("Reading APWSJ dataset, Errors : ", errors)

    docs_json = {k.strip(): v.strip() for k, v in docs_json.items()}

    topic_to_doc_file = "./dataset/apwsj/NoveltyData/apwsj.qrels"
    with open(topic_to_doc_file, "r") as f:
        topic_to_doc = f.read()
    topic_doc = [
        (i.split(" 0 ")[1][:-2], i.split(" 0 ")[0]) for i in topic_to_doc.split("\n")
    ]
    topics = "q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
    topic_list = topics.split(", ")
    filterd_docid = [(k, v) for k, v in topic_doc if v in topic_list]

    def crawl(red_dict, doc, crawled):
        ans = []
        for cdoc in red_dict[doc]:
            ans.append(cdoc)
            if crawled[cdoc] == 0:
                try:
                    red_dict[cdoc] = crawl(red_dict, cdoc, crawled)
                    crawled[cdoc] = 1
                    ans += red_dict[cdoc]
                except:
                    crawled[cdoc] = 1
        return ans

    wf = "./dataset/apwsj/redundancy_list_without_partially_redundant.txt"
    topics_allowed = "q101, q102, q103, q104, q105, q106, q107, q108, q109, q111, q112, q113, q114, q115, q116, q117, q118, q119, q120, q121, q123, q124, q125, q127, q128, q129, q132, q135, q136, q137, q138, q139, q141"
    topics_allowed = topics_allowed.split(", ")
    red_dict = dict()
    allow_partially_redundant = 1
    for line in open("./dataset/apwsj/NoveltyData/redundancy.apwsj.result", "r"):
        tokens = line.split()
        if tokens[2] == "?":
            if allow_partially_redundant == 1:
                red_dict[tokens[0] + "/" + tokens[1]] = [
                    tokens[0] + "/" + i for i in tokens[3:]
                ]
        else:
            red_dict[tokens[0] + "/" + tokens[1]] = [
                tokens[0] + "/" + i for i in tokens[2:]
            ]
    crawled = defaultdict(int)
    for doc in red_dict:
        if crawled[doc] == 0:
            red_dict[doc] = crawl(red_dict, doc, crawled)
            crawled[doc] = 1
    with open(wf, "w") as f:
        for doc in red_dict:
            if doc.split("/")[0] in topics_allowed:
                f.write(
                    " ".join(doc.split("/") + [i.split("/")[1] for i in red_dict[doc]])
                    + "\n"
                )

    write_file = "./dataset/apwsj/novel_list_without_partially_redundant.txt"
    topics = topic_list
    doc_topic_dict = defaultdict(list)

    for i in topic_doc:
        doc_topic_dict[i[0]].append(i[1])
    docs_sorted = (
        open("./dataset/apwsj/NoveltyData/apwsj88-90.rel.docno.sorted", "r")
        .read()
        .splitlines()
    )
    sorted_doc_topic_dict = defaultdict(list)
    for doc in docs_sorted:
        if len(doc_topic_dict[doc]) > 0:
            for t in doc_topic_dict[doc]:
                sorted_doc_topic_dict[t].append(doc)
    redundant_dict = defaultdict(lambda: defaultdict(int))
    for line in open(
        "./dataset/apwsj/redundancy_list_without_partially_redundant.txt", "r"
    ):
        tokens = line.split()
        redundant_dict[tokens[0]][tokens[1]] = 1
    novel_list = []
    for topic in topics:
        if topic in topics_allowed:
            for i in range(len(sorted_doc_topic_dict[topic])):
                if redundant_dict[topic][sorted_doc_topic_dict[topic][i]] != 1:
                    if i > 0:
                        # take at most 5 latest docs in case of novel
                        novel_list.append(
                            " ".join(
                                [topic, sorted_doc_topic_dict[topic][i]]
                                + sorted_doc_topic_dict[topic][max(0, i - 5) : i]
                            )
                        )
    with open(write_file, "w") as f:
        f.write("\n".join(novel_list))

    # Novel cases
    novel_docs = "./dataset/apwsj/novel_list_without_partially_redundant.txt"
    with open(novel_docs, "r") as f:
        novel_doc_list = [i.split() for i in f.read().split("\n")]
    # Redundant cases
    red_docs = "./dataset/apwsj/redundancy_list_without_partially_redundant.txt"
    with open(red_docs, "r") as f:
        red_doc_list = [i.split() for i in f.read().split("\n")]
    red_doc_list = filter(lambda x: len(x) > 0, red_doc_list)
    novel_doc_list = filter(lambda x: len(x) > 0, novel_doc_list)

    dataset = []
    s_not_found = 0
    t_not_found = 0
    for i in novel_doc_list:
        target_id = i[1]
        source_ids = i[2:]
        if target_id in docs_json.keys():
            data_inst = {}
            data_inst["target"] = docs_json[target_id]
            data_inst["source"] = ""
            for source_id in source_ids:
                if source_id in docs_json.keys():
                    data_inst["source"] += docs_json[source_id] + ". \n"
            data_inst["label"] = 1
        else:
            print(target_id)
        if data_inst["source"] != "":
            dataset.append(data_inst)

    for i in red_doc_list:
        target_id = i[1]
        source_ids = i[2:]
        if target_id in docs_json.keys():
            data_inst = {}
            data_inst["target"] = docs_json[target_id]
            data_inst["source"] = ""
            for source_id in source_ids:
                if source_id in docs_json.keys():
                    data_inst["source"] += docs_json[source_id] + ". \n"
            data_inst["label"] = 0
        else:
            print(target_id)
        if data_inst["source"] != "":
            dataset.append(data_inst)

    dataset_json = {}
    for i in range(len(dataset)):
        dataset_json[i] = dataset[i]

    dataset_path = "./dataset/apwsj/apwsj_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_json, f)
