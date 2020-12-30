
Glove=$1
sudo apt-get install subversion
pip3 install -r requirements.txt
mkdir -p ./dataset/novelty/
mkdir -p ./dataset/snli/
mkdir -p ./dataset/novelty/webis/
mkdir -p ./dataset/novelty/dlnd/
mkdir -p ./dataset/imdb/

wget "http://www.cs.cmu.edu/~yiz/research/NoveltyData/CMUNRF1.tar"
wget "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
wget "https://zenodo.org/record/3251771/files/Webis-CPC-11.zip"
wget "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

mv ./aclImdb_v1.tar.gz ./dataset/imdb/
tar -xf ./dataset/imdb/aclImdb_v1.tar.gz -C ./dataset/imdb/

Glove=$1
if [ $Glove == "1" ]; then
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip ./glove.840B.300d.zip 
else
    echo not downloading glove
fi

python3 scripts/setup.py $Glove


rm -rf Webis-CPC-11.zip
rm -rf snli_1.0.zip
rm -rf glove_vec.txt
rm -rf glove.840B.300d.zip
rm -rf glove.840B.300d.txt
rm -rf dlnd.zip
rm -rf CMUNRF1.tar

# python3 utils/download_models.py --neptune