FROM tensorflow/tfx:0.22.0

LABEL maintainer="hannes.hapke@gmail.com"

RUN CFLAGS=$(/usr/bin/python3.6-config --cflags) python3.6 -m pip install \
  "tensorflow-hub"

ENTRYPOINT ["python3.6", "/tfx-src/tfx/scripts/run_executor.py"]
