# Building Machine Learning Pipelines

Code repository for the O'Reilly publication ["Building Machine Learning Pipelines"](http://www.buildingmlpipelines.com) by Hannes Hapke &amp; Catherine Nelson

## Set up the demo project

Download the initial dataset. From the root of this repository, execute

```
python3 utils/download_dataset.py
```

After this script runs, you should have a `data` folder containing the file `consumer_complaints_with_narrative.csv`. The original source of this dataset is https://www.kaggle.com/cfpb/us-consumer-finance-complaints?select=consumer_complaints.csv

## Pre-pipeline experiment

## Interactive pipeline

The `interactive-pipeline` folder contains a full interactive TFX pipeline for the consumer complaint data.

## Full pipelines with Apache Beam, Apache Airflow, Kubeflow Pipelines, GCP

## Chapters

### Data privacy
Chapter 14. Code for training a differentially private version of the demo project. Note that the TF-Privacy module only supports TF 1.x as of June 2020.
