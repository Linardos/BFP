# Use nvidia/cuda image
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3-pip

# Install software 
RUN apt-get install -y git
RUN apt install python3-pip

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN git clone https://github.com/Linardos/BFP 
#RUN cd src
WORKDIR "/BFP"
#RUN pip install virtualenv
RUN apt install python3.8-venv
RUN python3 -m venv fl_env 
RUN source fl_env/bin/activate 
RUN pwd

RUN pip install -r /src/requirements.txt 
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
RUN apt-get install ffmpeg libsm6 libxext6 -y 
RUN export PYTHONPATH=~/BFP 
RUN touch __init__.py 

#WORKDIR /BFP
COPY ./*.yaml ./
COPY ./*.py ./
COPY ./models ./models
COPY ./preprocessing ./preprocessing
COPY ./data_augmentation ./data_augmentation
COPY ./data_handling ./data_handling
COPY ./requirements.txt ./requirements.txt
COPY ./logs ./logs

RUN export PYTHONPATH=~/BFP
RUN touch __init__.py
#RUN python3 client.py

