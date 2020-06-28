## How to use TFX templates

### Download the base template

You can download the taxi example template (at the moment, the only available template) and update the individual files. We have done this for our _complaint prediction_ example pipeline.

We ran:
```
$ tfx template copy --pipeline_name complaint_prediction_pipeline --destination_path . --model=taxi
```

### Test your pipeline

Add this path to your `PYTHONPATH` with:

```
$ cd building-machine-learning-pipelines/chapters/appendix_c/tfx_template_example
$ export PYTHONPATH=$PYTHONPATH:`pwd`
```

Afterward, you can run the tests with:
```
$ pytest . -s
```

Note: The first test might take some time, since the TF Hub model will be downloaded.
