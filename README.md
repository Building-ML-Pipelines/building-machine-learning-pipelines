# Building Machine Learning Pipelines

Code repository for the O'Reilly publication ["Building Machine Learning Pipelines"](http://www.buildingmlpipelines.com) by Hannes Hapke &amp; Catherine Nelson

## Set up the demo project

Download the initial dataset. From the root of this repository, execute

```
python3 utils/download_dataset.py
```

After this script runs, you should have a `data` folder containing the file `consumer_complaints_with_narrative.csv`. 

## The dataset

The data that we use in this example project can be downloaded using the script above. The dataset is from a public dataset on customer complaints collected from the US Consumer Finance Protection Bureau. If you would like to reproduce our edited dataset, carry out the following steps:

- Download the dataset from https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
- Rename the columns to `[
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "consumer_complaint_narrative",
        "company",
        "state",
        "zip_code",
        "company",
        "company_response",
        "timely_response",
        "consumer_disputed"]`
- Filter the dataset to remove rows with missing data in the `consumer_complaint_narrative` column
- In the `consumer_disputed` column, map `Yes` to `1` and `No` to `0`


## Pre-pipeline experiment

Before building our TFX pipeline, we experimented with different feature engineering and model architectures. The notebooks in this folder preserve our experiments, and we then refactored our code into the interactive pipeline below.

## Interactive pipeline

The `interactive-pipeline` folder contains a full interactive TFX pipeline for the consumer complaint data.

## Full pipelines with Apache Beam, Apache Airflow, Kubeflow Pipelines, GCP

The `pipelines` folder contains complete pipelines for the various orchestrators. See Chapters 11 and 12 for full details.

## Chapters

The following subfolders contain stand-alone code for individual chapters.

### Model analysis
Chapter 7. Stand-alone code for TFMA, Fairness Indicators, What-If Tool. Note that these notebooks will not work in JupyterLab.

### Data privacy
Chapter 14. Code for training a differentially private version of the demo project. Note that the TF-Privacy module only supports TF 1.x as of June 2020.
