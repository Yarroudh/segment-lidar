FROM ubuntu:22.04

RUN useradd --create-home --shell /bin/bash user
WORKDIR /home/user

# Install Anaconda
RUN apt-get update && apt-get install -y wget bzip2
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc

# Configure the environment
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n samlidar python=3.9
RUN echo "conda activate samlidar" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
USER root

# Install GCC
RUN apt-get install gcc -y

# Install segment-lidar
RUN python -m pip install --upgrade pip setuptools wheel cython
RUN python -m pip install segment-lidar

CMD ["python", "-c", "while True: pass"]