## APWSJ
# %%
without_part_red = ".data/apwsj/trec/CMUNRF/novel_list_without_partially_redundant.txt"
with_part_red = ".data/apwsj/trec/CMUNRF/redundancy_list_without_partially_redundant.txt"
with open(without_part_red,'r') as f:
    without_part_red_data = f.read()

with open(with_part_red,'r') as f:
    with_part_red_data = f.read()
# %%
import re
ids = re.findall('(AP\d+-\d+|WSJ\d+-\d+)',with_part_red_data+"\n"+with_part_red_data)
# %%
ids = set(ids)
print(len(ids))
# %%
import os
import glob
path = '.data/apwsj/trec'
AP_path = os.path.join(path, "AP")
AP_files = glob.glob(os.path.join(AP_path, "*.gz"))
for i in AP_files:
    with gzip.open(i, "r") as f:
        text = f.read()

    with open(i[:-3], "wb") as f_new:
        f_new.write(text)
    os.remove(i)

wsj = os.path.join(path, "TREC", "wsj")
ap = os.path.join(path, "TREC", "ap")
ap_others = os.path.join(path, "AP")
wsj_others = os.path.join(path, "WSJ", "wsj_split")
cmunrf = os.path.join(path, "CMUNRF")

wsj_files = glob.glob(wsj + "/*")
ap_files = glob.glob(ap + "/*")

wsj_other_files = []
ap_other_files = glob.glob(os.path.join(ap_others, "*"))

wsj_big = glob.glob(os.path.join(wsj_others, "*"))
for i in wsj_big:
    for file_path in glob.glob(os.path.join(i, "*")):
        wsj_other_files.append(file_path)
errors = 0
# %%

##WSJ in TREC
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
wsj_trec_ids = docs_json.keys()
wsj_trec_ids = [i.strip() for i in wsj_trec_ids]
print(len(ids.intersection(set(list(wsj_trec_ids)))), "in SVN/wsj")
for i in ids.intersection(set(list(wsj_trec_ids))):
    ids.remove(i)
#  378 in ids
# %%
docs_json = {}
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

ap_trec_ids = docs_json.keys()
ap_trec_ids = [i.strip() for i in ap_trec_ids]
print(len(ids.intersection(set(list(ap_trec_ids)))), "in SVN/ap")
for i in ids.intersection(set(list(ap_trec_ids))):
    ids.remove(i)
# %%
docs_json = {}
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


ap_other_trec_ids = docs_json.keys()
ap_other_trec_ids = [i.strip() for i in ap_other_trec_ids]
print(len(ids.intersection(set(list(ap_other_trec_ids)))), "in AP Other")
for i in ids.intersection(set(list(ap_other_trec_ids))):
    ids.remove(i)
# %%
docs_json = {}
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

wsj_split_trec_ids = docs_json.keys()
wsj_split_trec_ids = [i.strip() for i in wsj_split_trec_ids]
print(len(ids.intersection(set(list(wsj_split_trec_ids)))), "in wsj_split")
for i in ids.intersection(set(list(wsj_split_trec_ids))):
    ids.remove(i)
# %%
a="""WSJ880406-0066
WSJ880504-0084
WSJ880504-0110
WSJ880505-0011
WSJ880505-0116
WSJ880819-0038
WSJ880819-0090
WSJ880822-0013
WSJ880822-0026
WSJ880822-0108
WSJ880823-0003
WSJ880823-0117
WSJ880824-0050
WSJ880824-0087
WSJ880825-0035
WSJ880927-0063
WSJ880927-0114
WSJ880928-0083
WSJ880928-0085
WSJ880928-0091
WSJ890905-0051
WSJ890905-0079
WSJ890905-0126
WSJ890906-0010
WSJ890906-0134
WSJ890918-0117
WSJ890918-0123
WSJ890918-0173
WSJ890919-0054
WSJ890919-0141
WSJ890920-0011"""
a = a.split("\n")
# %%
for i in a:
    if i in ids:
        print(i)
# %%
