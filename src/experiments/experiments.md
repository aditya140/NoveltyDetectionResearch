
# List of experiments performed

- [ ]   10 Fold Validation
- [ ]   Impact of Sentence Encoder pretraining 
- [ ]   Impact of Document Encoder pretraining 
- [ ]   Number of Samples Required to learn the concept of Novelty


## 10 Fold Validation 

We perform 10 fold vaildation test on 3 dataset - 
1. TAP-DLND
2. APWSJ (Scraped/Partial)
3. Webis-CPC

### TAP DLND

| **MODEL**           |   Non Novel Precision |   Novel Precision |   Non Novel Recall |   Novel Recall |   Non Novel F1 |   Novel F1 |   Accuracy |
| :---                |         :----:        |      :----:       |       :----:       |      :----:    |    :----:      |   :----:   |       ---: |
| tfidf_novelty_score |              0.515915 |          0        |           1        |       0        |       0.680665 |   0        |   51.5915  |
| set_diff            |              0.554823 |          0.546018 |           0.660485 |       0.546018 |       0.603061 |   0.484349 |   55.1426  |
| geo_diff            |              0.740084 |          0.657423 |           0.625535 |       0.657423 |       0.678005 |   0.707514 |   69.3468  |
| kl_div              |              0.740487 |          0.744241 |           0.770328 |       0.744241 |       0.755113 |   0.727908 |   74.2226  |
| pv                  |              0.751045 |          0.747561 |           0.769258 |       0.747561 |       0.760042 |   0.737774 |   74.9402  |
| dan                 |              0.865656 |          0.882933 |           0.894306 |       0.882933 |       0.879748 |   0.867041 |   87.3713  |
| han                 |              0.874518 |          0.878237 |           0.8879   |       0.878237 |       0.881158 |   0.870999 |   87.6287  |
| ein                 |              0.869535 |          0.887968 |           0.898932 |       0.887968 |       0.88399  |   0.871636 |   87.8125  |
| rdv_cnn             |              0.910969 |          0.904255 |           0.91032  |       0.904255 |       0.910644 |   0.904599 |   90.7721  |


### APWSJ(Scraped/Partial)

| **MODEL**           |   Non Novel Precision |   Novel Precision |   Non Novel Recall |   Novel Recall |   Non Novel F1 |   Novel F1 |   Accuracy |
| :---                |         :----:        |      :----:       |       :----:       |      :----:    |    :----:      |   :----:   |       ---: |

### Webis-CPC

| **MODEL**           |   Non Novel Precision |   Novel Precision |   Non Novel Recall |   Novel Recall |   Non Novel F1 |   Novel F1 |   Accuracy |
| :---                |         :----:        |      :----:       |       :----:       |      :----:    |    :----:      |   :----:   |       ---: |



## Impact of Sentence Encoder pretraining
We test the impact of sentence encoder pretraining on our Hierarchical Self-Attention Network by using 5 random runs for each type of sentence encoder with a 80-10-10 split of the dataset(Novelty Detection)

We use the following sentence encoders - 
1. BiLSTM Encoder + Spacy
2. BiLSTM Encoder + WordPiece
3. BiLSTM + Self Attention Encoder + Spacy
4. BiLSTM + Self Attention Encoder + WordPiece

### Accuracy on NLI tasks

| Model | Tokenizer | SNLI Test Accuracy | MNLI Test Accuracy|
| :---  |    :---:  |             :---:  |             ---:  |
| BiLSTM| Spacy     | - | -|
| BiLSTM| Spacy     | - | -|
| BiLSTM + self attention | WordPiece     | - | -|
| BiLSTM + self attention | WordPiece     | - | -|



## Impact of Document Encoder pretraining
We use the sentence encoder which provides the best accuracy as the sentence encoder for the document encoder.

We test the impact of sentence encoder pretraining on our Hierarchical Self-Attention Network by using 5 random runs for each type of sentence encoder with a 80-10-10 split of the dataset(Novelty Detection)

We use the following sentence encoders - 
1. BiLSTM Encoder + Spacy
2. BiLSTM Encoder + WordPiece
3. BiLSTM + Self Attention Encoder + Spacy
4. BiLSTM + Self Attention Encoder + WordPiece

