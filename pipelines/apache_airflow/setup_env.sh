# Set up the environment for the TFX tutorial
# Adopted from the TFX setup

GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

# printf "${GREEN}Installing TFX workshop${NORMAL}\n\n"
# # TF/TFX prereqs
# printf "${GREEN}Installing TensorFlow${NORMAL}\n"
# pip install tensorflow==2.1.0

# printf "${GREEN}Installing TFX${NORMAL}\n"
# pip install tfx==0.21.3

printf "${GREEN}Installing Google API Client${NORMAL}\n"
pip install google-api-python-client

# printf "${GREEN}Installing required Jupyter version${NORMAL}\n"
# pip install ipykernel
# ipython kernel install --user --name=tfx
# pip install --upgrade notebook
# jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis
# jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis

# printf "${GREEN}Installing packages used by the notebooks${NORMAL}\n"
# pip install matplotlib
# pip install papermill
# pip install pandas
# pip install networkx

# jupyter nbextension enable tensorflow_model_analysis --py --sys-prefix

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
airflow initdb

exit
