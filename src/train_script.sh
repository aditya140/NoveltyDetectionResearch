pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds True --scheduler step --optim adam --epochs 10 han &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds True --scheduler step --optim adam --epochs 10 dan &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds True --scheduler step --optim adam --epochs 10 rdv_cnn &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds True --scheduler step --optim adam --epochs 10 diin &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds True --scheduler step --optim adam --epochs 10 adin &&


pip3 install -r requirements.txt &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-87 --seed 1029 --folds True --scheduler step --optim adam --epochs 10 han


pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds --epochs 10 struc --attention_hops 5 --attention_layer_param 200 --prune_p 150 --prune_p 15 &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 struc --attention_hops 10 --attention_layer_param 200 --prune_p 150 --prune_p 15 &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 struc --attention_hops 20 --attention_layer_param 200 --prune_p 150 --prune_p 15 &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 struc --attention_hops 40 --attention_layer_param 200 --prune_p 150 --prune_p 15 &&



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --seed 1029 --folds True --epochs 10 struc --attention_hops 25 --attention_layer_param 200 --prune_p 200 --prune_p 20



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/tune/tune_novelty.py -d dlnd --load_nli NLI-93 --num_trials 40 --epochs 8 --sampler grid dan



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/tune/tune_novelty.py -d dlnd --load_nli NLI-93 --num_trials 40 --epochs 8 --sampler grid mwan



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/tune/tune_novelty.py -d dlnd --load_nli NLI-93 --num_trials 40 --epochs 8 struc




pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/tune/tune_novelty.py -d dlnd --load_nli NLI-93 --num_trials 40 --epochs 8 adin



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/tune/tune_novelty.py -d dlnd --load_nli NLI-93 --num_trials 40 --epochs 8 han


pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/tune/tune_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 --num_trials 100 struc



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --freeze_encoder True --epochs 10 han &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 han &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --freeze_encoder True --epochs 10 dan &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 dan &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --freeze_encoder True --epochs 10 rdv_cnn &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 rdv_cnn &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --freeze_encoder True --epochs 10 adin &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 adin &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --freeze_encoder True --epochs 10 struc &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 struc



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 han &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 dan &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 rdv_cnn &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 adin &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 struc




pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/ensemble/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 10 dan



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/ensemble/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 8 struc



pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/ensemble/train_novelty.py -d dlnd --load_nli NLI-93 --epochs 8 adin


python3 src/train/train_nli.py -d mnli --tokenizer spacy bilstm
python3 src/train/train_nli.py -d mnli --tokenizer spacy attention
python3 src/train/train_nli.py -d mnli --tokenizer spacy struc_attn
python3 src/train/train_nli.py -d mnli --tokenizer spacy mwan



python3 src/train/train_nli.py -d mnli --tokenizer distil_bert --max_len 80 --epochs 10 --loss_agg mean bert


