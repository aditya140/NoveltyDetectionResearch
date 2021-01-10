FROM nvidia/cuda:10.2-devel-ubuntu18.04
RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN apt-get install -y wget build-essential python-dev git
RUN pip3 install pip -U && pip3 install setuptools -U && pip3 install -U spacy
RUN useradd -m aditya
RUN chown -R aditya:aditya /home/aditya/
COPY --chown=aditya . /home/aditya/
USER aditya
RUN cd /home/aditya && bash setup.sh GLOVE=0 NOV_DATA=1 DOC_DATA=0 SNLI_DATA=0
WORKDIR /home/aditya
