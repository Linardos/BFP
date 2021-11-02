# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN git clone https://github.com/Linardos/BFP 
RUN cd src
RUN pip install virtualenv
RUN python3 -m venv fl_env 
RUN source fl_env/bin/activate 
RUN pip install -r requirements.txt 
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
RUN export PYTHONPATH=~/BFP 
RUN touch __init__.py 

#WORKDIR /src
COPY ./*.yaml ./
COPY ./*.py ./
COPY ./models ./models
COPY ./preprocessing ./preprocessing
COPY ./data_augmentation ./data_augmentation
COPY ./data_handling ./data_handling
COPY ./requirements.txt ./requirements.txt


RUN ls
CMD ["python", "client.py"]

