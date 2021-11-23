# Building Machine Learning Pipelines

Code repository for the O'Reilly publication ["Building Machine Learning Pipelines"](http://www.buildingmlpipelines.com) by Hannes Hapke &amp; Catherine Nelson

## Update

* The example code has been updated to work with TFX 1.4.0, TensorFlow 2.6.1, and Apache Beam 2.33.0. A GCP Vertex example (training and serving) was added.

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

### Advanced TFX
Chapter 10. Notebook outlining the implementation of custom TFX components from scratch and by inheriting existing functionality. Presented at the Apache Beam Summit 2020.

### Data privacy
Chapter 14. Code for training a differentially private version of the demo project. Note that the TF-Privacy module only supports TF 1.x as of June 2020.

## Version notes

The code was written and tested for version 0.22.

- As of 11/23/21, the examples have been updated to support TFX 1.4.0, TensorFlow 2.6.1, and Apache Beam 2.33.0. A GCP Vertex example (training and serving) was added.

- As of 9/22/20, the interactive pipeline runs on TFX version 0.24.0rc1.
Due to tiny TFX bugs, the pipelines currently don't work on the releases 0.23 and 0.24-rc0. Github issues have been filed with the TFX team specifically for the book pipelines ([Issue 2500](https://github.com/tensorflow/tfx/issues/2500#issuecomment-695363847)). We will update the repository once the issue is resolved.

- As of 9/14/20, TFX only supports Python 3.8 with version >0.24.0rc0.
