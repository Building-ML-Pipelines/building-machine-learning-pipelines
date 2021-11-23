# Set up the environment for the TFX tutorial
# Adopted from the TFX setup

GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

printf "${GREEN}Installing Google API Client${NORMAL}\n"
pip install google-api-python-client

# # Docker images
printf "${GREEN}Installing docker${NORMAL}\n"
pip install docker

# Airflow
# Set this to avoid the GPL version; no functionality difference either way
printf "${GREEN}Preparing environment for Airflow${NORMAL}\n"
export SLUGIFY_USES_TEXT_UNIDECODE=yes
printf "${GREEN}Installing Airflow${NORMAL}\n"
pip install apache-airflow
printf "${GREEN}Initializing Airflow database${NORMAL}\n"
airflow db init
airflow users create --username "tfx" \
                     --firstname "TensorFlow" \
                     --lastname "Extended" \
                     --role "Admin" \
                     --email "admin@example.org" \
                     --password "tfx"
