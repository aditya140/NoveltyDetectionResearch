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
create_error_report("dan_analysis.csv")

# %%
create_error_report("cnn_analysis.csv")
# %%
import pandas as pd

han_cnn_rep = pd.read_csv("han_cnn_analysis_report.csv")
han_rep = pd.read_csv("han_analysis_report.csv")
dan_rep = pd.read_csv("dan_analysis_report.csv")
cnn_rep = pd.read_csv("cnn_analysis_report.csv")

# %%

han_analysis = pd.read_csv("han_analysis.csv")
han_cnn_analysis = pd.read_csv("han_cnn_analysis.csv")
dan_analysis = pd.read_csv("dan_analysis.csv")
cnn_analysis = pd.read_csv("cnn_analysis.csv")

# %%
def intersection_ids(a,b):
    return set(a.id).intersection(b.id)

# %%
print("HAN_CNN errors - ",len(han_cnn_rep.id) )
print("DAN errors - ",len(dan_rep.id) )
print("HAN errors - ",len(han_rep.id) )
print("CNN errors - ",len(cnn_rep.id) )
# %%

print("HAN - CNN common errors - ",len(intersection_ids(han_rep,cnn_rep)) )
print("HAN - DAN common errors - ",len(intersection_ids(han_rep,dan_rep)) )
print("HAN - HAN_CNN common errors - ",len(intersection_ids(han_rep,han_cnn_rep)) )
print("DAN - CNN common errors - ",len(intersection_ids(dan_rep,cnn_rep)) )
print("DAN - HAN_CNN common errors - ",len(intersection_ids(dan_rep,han_cnn_rep)) )
print("HAN_CNN - CNN common errors - ",len(intersection_ids(han_cnn_rep,cnn_rep)) )
# %%

import nltk

def report_analysis(rep):
    target_num_words = rep['target_text'].apply(lambda x: len(nltk.word_tokenize(x)))
    target_num_sent = rep['target_text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    source_num_words = rep['source'].apply(lambda x: len(nltk.word_tokenize(x)))
    source_num_sent = rep['source'].apply(lambda x: len(nltk.sent_tokenize(x)))
    analysis = pd.DataFrame({"target_num_words":target_num_words,"target_num_sent":target_num_sent,"source_num_words":source_num_words,"source_num_sent":source_num_sent})
    return analysis.describe(percentiles=[])
# %%
report_analysis(han_rep)

# %%
report_analysis(han_cnn_rep)
# %%
report_analysis(dan_rep)
# %%

report_analysis(cnn_rep)
# %%
# HAN - HAN_CNN common errors
report_analysis(han_rep.merge(han_cnn_rep))
# %%
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

# %%



len(nltk.sent_tokenize('''New Delhi: A day after Delhi Chief Arvind Kejriwal "saluted" Prime Minister Narendra Modi for the 'surgical strikes' by the Army on terror launch pads across the LoC, the Opposition on Tuesday demanded the NDA government to give adequate proof of its much-hyped military action against Pakistan.

UPA did PoK strikes too, it is up to the present govt to give proof of the surgical strikes, former home minister P Chidambaram told News18.

The much-hyped surgical strike on last Wednesday night was not the only time that Army had crossed the LoC to take punitive action, the former home minister said.

A similar 'major strike' took place in January 2013, when the UPA was in power, he added. Chidambaram further said that the then government chose not to go public "in keeping with its policy of strategic restraint ".

The Congress leader, however, cautioned the NDA government against taking political ownership of the surgical strikes and drawing any premature conclusions from the same.

When quizzed about the current deadlock between India and Pakistan, Chidambaram said, ''How long can we stay away from dialogue and non-political ties.''

''No cricket, no films is not a strategy to adopt,'' he added.

While political parties, cutting across their ideological divide, have expressed support to the Narendra Modi government for giving a go ahead to the armed forces for conducting surgical strikes across the LoC against terror launch pads backed by Pakistan, the opposition has also demanded the government to give proof of the military action.

Chidambaram's remarks came a day after Kejriwal, a noted critic of PM Modi, urged the Centre to counter Pakistan's smear campaign on international stage.

Kejriwal, who is often at loggerheads with the Centre, said he may have differences with the Prime Minister over several issues, but by undertaking the surgical strikes, Modi has shown the will to deal with Pakistan.

"Last week our army showed valour and avenged the deaths of 19 soldiers killed in the Uri attack. I may have differences with the Prime Minister over a 100 issues. But when he has shown the will (to deal with this matter), I salute him," he said.

This is perhaps the first time that Kejriwal, who has been critical about the Modi government and its Pakistan policy, has come out praising him. On the day of the strikes, the Delhi Chief Minister had hailed the army, but there was no word of praise for Modi.

A day later, he had told the Delhi Assembly that is time to stand with the Centre and the differences between them can be sorted out later.

Claiming that Pakistan has gone "berserk" after the strike, Kejriwal said, it is resorting to smear campaign against India at international fora and this has to be countered.

"It has resorted to playing dirty politics. Since the last two days, Pakistan is taking international journalists to the border and trying to show that surgical strikes never took place.

"Two days back, the United Nations gave a statement that there was no such activity on the border," he said.

"My blood boiled over these news reports (of the international media). Pakistan is indulging in smearing India's image at the international level.

The UN Military Observer Group in India and Pakistan (UNMOGIP) "has not directly observed" any firing along the LoC, UN chief Ban Ki-moon's spokesperson  Stephane Dujarric had said on September 30 against the backdrop of the surgical strikes conducted by India.

"I appeal to the Prime Minister that the way he and the Army taught Pakistan a lesson on ground, he should also unmask the propaganda by Pakistan at international level. The whole country is with you. I also appeal to the countrymen not to believe in the false campaign by Pakistan," he said.'''))
# %%
