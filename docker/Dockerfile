# Use nvidia/cuda image
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3-pip git

ARG DEBIAN_FRONTEND=noninteractive

#RUN git clone https://github.com/Linardos/BFP 
WORKDIR "/BFP/docker"
#RUN cd src
#RUN pip install virtualenv
RUN apt install python3.8-venv
RUN python3 -m venv fl_env 

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
RUN source fl_env/bin/activate 
RUN pwd

COPY . /BFP/docker
RUN pip3 install -r requirements.txt
RUN pip3 install --timeout 1000 torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir /BFP/dataset
ENV PYTHONPATH "${PYTHONPATH}:/BFP"

WORKDIR "/BFP/src"
#RUN python3 client.py
CMD [ "python3", "client.py" ]
#ENTRYPOINT [ "python3" ]

