# Inference

## Parallelizing inference

When processing multiple documents, we can optimize the inference by parallelizing the
computation on multiple cores and GPUs.

Assuming we have parquet files that follow the
[schema described in the quickstart page](../quickstart/#apply-on-multiple-documents),
we can use the following code to parallelize the processing of the documents. For a deep
learning model, the idea is to have enough CPU workers, which manage the IO, rule-based
components and torch pre- / post-processing, to pre-process batches of documents in parallel
to the deep learning model computation in order to maximise GPU utilisation.

We will output the result to a dataset parquet folder, with 8192 documents per file. This
dataset can be located on the filesystem, or on a distributed filesystem like HDFS or S3.

```python
import edsnlp

data = edsnlp.data.read_parquet("path/to/parquet_folder", converter=converter)
data = data.map_pipeline(nlp)
data = data.set_processing(
    # 1 GPUs to accelerate deep-learning pipes
    num_gpu_workers=1,  # or more if you have more GPUs
    # 4 CPUs for rb pipes, IO and pre- / post-processing
    num_cpu_workers=4,
    # Each worker will process 32 docs at a time
    batch_size=32,
    # Track the progress of the processing
    show_progress=True,
)
data.write_parquet(
    "hdfs://path/to/output_folder",
    converter="ents",
    # Each file will contain the annotations of 8192 docs
    num_rows_per_file=8192,
    # Each worker will write directly to the output folder
    # without
    write_in_worker=True,
)
```

All the parameters of the `set_processing` method can be found in the
[API documentation](https://aphp.github.io/edsnlp/latest/concepts/inference/#edsnlp.core.lazy_collection.LazyCollection.set_processing).

Below are some tips to help you choose the right number of workers and batch size. You
should multiply the number of CPU workers `num_cpu_workers` by the number of GPUs to get
the total number of CPU workers. These numbers are only indicative, and you should take
into account:

- the average size of your documents (bigger texts means smaller batch size), these numbers
  are for documents averaging 2200 characters
- number of available cores: ensure that `num_cpu_workers` + `num_gpu_workers` < number of cores
- the number of rule-based pipes : if you have a lot of rule-based pipes, you should
  increase the number of CPU workers. These numbers assume that you have disabled the
  `dates` and `simpler_rules/PERSON` patterns.

Make sure to monitor the GPU usage with `nvidia-smi` or `watch -n 5 nvidia-smi` to make
sure your GPU(s) are fully utilized.

| GPU model  | `batch_size` | `num_cpu_workers` per gpu |
|------------|--------------|---------------------------|
| ~80Gb A100 | < 128        | 6-8                       |
| ~32Gb V100 | < 48         | 3-4                       |
| ~24Gb P40  | < 32         | 1-2                       |
