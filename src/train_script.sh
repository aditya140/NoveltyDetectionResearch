pip3 install -r requirements.txt &&
python3 -m spacy download en &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-87 --epochs 10 --seed True --folds True --scheduler step --optim adam --epochs 10 han &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-87 --epochs 10 --seed True --folds True --scheduler step --optim adam --epochs 10 dan &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-87 --epochs 10 --seed True --folds True --scheduler step --optim adam --epochs 10 rdv_cnn &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-87 --epochs 10 --seed True --folds True --scheduler step --optim adam --epochs 10 diin &&
python3 src/train/train_novelty.py -d dlnd --load_nli NLI-87 --epochs 10 --seed True --folds True --scheduler step --optim adam --epochs 10 adin &&


