
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            GLOVE)              GLOVE=${VALUE} ;;
            DOC_DATA)    DOC_DATA=${VALUE} ;;     
            NOV_DATA)    NOV_DATA=${VALUE} ;;     
            SNLI_DATA)    SNLI_DATA=${VALUE} ;;     
            *)   
    esac    

done

echo "GLOVE = $GLOVE"
echo "DOC_DATA = $DOC_DATA"
echo "NOV_DATA = $NOV_DATA"
echo "SNLI_DATA = $SNLI_DATA"



sudo apt-get install subversion
pip3 install -r requirements.txt

pip3 install kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json




if [ $DOC_DATA == "1" ]; then
    mkdir -p ./dataset/imdb/
    mkdir -p ./dataset/yelp/
    mkdir -p ./dataset/arxiv/
    mkdir -p ./dataset/reuters/
    wget "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    wget "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
    kaggle datasets download -d yelp-dataset/yelp-dataset
    kaggle datasets download -d Cornell-University/arxiv
    mv ./aclImdb_v1.tar.gz ./dataset/imdb/
    tar -xf ./dataset/imdb/aclImdb_v1.tar.gz -C ./dataset/imdb/
else
    echo not downloading Document Classification Data
fi


if [ $NOV_DATA == "1" ]; then
    mkdir -p ./dataset/novelty/
    mkdir -p ./dataset/novelty/webis/
    mkdir -p ./dataset/novelty/dlnd/
    wget "http://www.cs.cmu.edu/~yiz/research/NoveltyData/CMUNRF1.tar"
    wget "https://zenodo.org/record/3251771/files/Webis-CPC-11.zip"

else
    echo not downloading Novelty Data
fi


if [ $SNLI_DATA == "1" ]; then
    mkdir -p ./dataset/snli/
    wget "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    
else
    echo not downloading SNLI
fi


if [ $GLOVE == "1" ]; then
    wget "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    unzip ./glove.840B.300d.zip 
else
    echo not downloading glove
fi

python3 scripts/setup.py --glove $GLOVE --novelty $NOV_DATA --document $DOC_DATA --snli $SNLI_DATA


rm -rf Webis-CPC-11.zip
rm -rf snli_1.0.zip
rm -rf glove_vec.txt
rm -rf glove.840B.300d.zip
rm -rf glove.840B.300d.txt
rm -rf dlnd.zip
rm -rf CMUNRF1.tar
rm -rf yelp-dataset.zip
rm -rf arxiv.zip
rm -rf reuters21578.tar.gz
rm -rf dataset_apw.zip

python3 utils/download_models.py --neptunes