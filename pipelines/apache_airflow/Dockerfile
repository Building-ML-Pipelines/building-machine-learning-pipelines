FROM gcr.io/tfx-oss-public/tfx:1.4.0

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

WORKDIR /root
RUN mkdir -p tfx
RUN mkdir -p airflow/dags

COPY setup_airflow.sh setup_airflow.sh
COPY setup_env.sh setup_env.sh
COPY launch_airflow.sh launch_airflow.sh

RUN sh setup_env.sh
RUN sh setup_airflow.sh
RUN chmod +x launch_airflow.sh

EXPOSE 8081
EXPOSE 7070

ENTRYPOINT [ "sh", "-c", "./launch_airflow.sh" ]
