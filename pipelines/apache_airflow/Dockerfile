FROM tensorflow/tfx:latest

LABEL maintainer="hannes.hapke@gmail.com"

RUN apt-get update
RUN apt-get install -y \
        curl \
        gnupg \
        vim \
        git \
        build-essential \
        tmux \
        htop

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV SLUGIFY_USES_UNIDECODE=yes

RUN cd /root; mkdir tfx; cd tfx
RUN mkdir -p /root/airflow/dags

# RUN git clone https://github.com/tensorflow/tfx.git
COPY setup_airflow.sh /root/setup_airflow.sh
COPY setup_env.sh /root/setup_env.sh
COPY launch_airflow.sh /root/launch_airflow.sh

WORKDIR /root/
RUN sh setup_env.sh
RUN sh setup_airflow.sh
RUN chmod +x launch_airflow.sh
ENTRYPOINT [ "sh", "-c", "./launch_airflow.sh" ]
