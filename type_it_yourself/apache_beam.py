import re
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions

input_file = "gs://dataflow-samples/shakespeare/kinglear.txt"
output_file = "/tmp/output.txt"

# Define pipeline options object.
pipeline_options = PipelineOptions()
with beam.Pipeline(
    options=pipeline_options
) as p:  # Read the text file or file pattern into a PCollection.
    lines = p | ReadFromText(input_file)

# Count the occurrences of each word.
counts = (
    lines
    | "Split" >> beam.FlatMap(lambda x: re.findall(r"[A-Za-z\']+", x))
    | "PairWithOne" >> beam.Map(lambda x: (x, 1))
    | "GroupAndSum" >> beam.CombinePerKey(sum)
)


# Format the counts into a PCollection of strings.
def format_result(word_count):
    (word, count) = word_count
    return "{}: {}".format(word, count)


output = counts | "Format" >> beam.Map(format_result)
# Write the output using a "Write" transform that has side effects.
output | WriteToText(output_file)

# python apache_beam.py
