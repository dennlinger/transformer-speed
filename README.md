# transformer-speed

Author: Dennis Aumiller, Heidelberg University

This repository provides a quick evaluation setup to benchmark wall clock timings of Transformer networks. 
Specifically, this also collects results from various systems.
Feel free to open a PR with timings with your local hardware,
to get a better understanding of how fast/slow inference on various systems is.
This specifically does not utilize `transformers.pipelines`, 
which make some assumption that do not benefit performance, 
specifically with respect to the input length and padding strategy. 

## Caveats
Currently, this only supports CPU execution, but I plan to incorporate GPU inference soon.
Note that the latter also suffers from overhead that comes from shuffling memory between the CPU/GPU memories,
which makes the results not directly comparable.
Especially when working with small batch sizes, the overhead of transfer might exceed the performance boost.
Further, existing computing infrastructure supports mainly CPU
(especially in smaller companies that don't rely on Cloud Service provides like AWS/GCP/Azure).

# Installation & Usage

To install this library, simply clone the repository, and run
```bash
python3 -m pip install -r requirements.txt
```
from the repository's directory.

To run the script, simply provide a model name and the desired sequence length (available up to the model's maximum; generally 512 tokens).
A standard run might look like this:
```bash
python3 main.py bert-base-uncased 256
```


To see all available arguments, have a look at `python3 main.py --help`, which will print the following output:

```
usage: main.py [-h] [--device DEVICE] [--batch_size BATCH_SIZE] [--seed SEED] [--num_samples NUM_SAMPLES] model_name max_length

positional arguments:
  model_name            The name of the Transformer model to use for this run.
  max_length            Maximum input length. Most transformer models restrict this to 512 subword tokens, however, they also benefit from a shorter input
                        length due to less calculations.

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       Which device to use for execution. Currently only supports CPU usage.
  --batch_size BATCH_SIZE
                        How many samples will be aggregated into a single forward pass. Does not affect the total number of --num_samples being processed
  --seed SEED           Random seed used by the numpy rng state to determine the samples
  --num_samples NUM_SAMPLES
                        Number of samples that should be processed during the entire process. Our data set currently provides ~200 inputs (of maximum
                        length), which will be chosen randomly, and potentially repeated.


```
# Results

Feel free to open a pull request with results from your local machine. It is always welcome to seen new system configurations for reference.

|Model                      | Sequence Length | Batch Size| Samples per second | `num_samples` | CPU                         | RAM        | Tokenization  | Inference |
| :------------------------ | --------------: | --------: | -----------------: | ------------: | :-------------------------- | :--------: | ------------: | --------: |
| `bert-base-uncased`       | 128             | 16        | 5.38               | 1000          | Intel(R) Core(TM) i7-7560U  | DDR3-1866  | 0.8039s       | 185.05s   |
|                           | 128             | 16        | 15.33              | 1000          | AMD Ryzen 5 5600X           | DDR4-3200  | 0.2826s       | 64.94s    |
|                           | 128             | 16        | 14.72              | 1000          | Intel(R) Xeon(R) Silver 4210| DDR4-2933  | 0.5708s       | 67.34s    |
|                           | 256             | 16        | 1.72               | 1000          | Intel(R) Core(TM) i7-7560U  | DDR3-1866  | 0.8476s       | 579.95s   |
|                           | 512             | 16        | 2.74               | 1000          | AMD Ryzen 5 5600X           | DDR4-3200  | 0.3065s       | 364.09s   |
| `distilbert-base-uncased` | 128             | 16        | 10.10              | 1000          | Intel(R) Core(TM) i7-7560U  | DDR3-1866  | 1.5550s       | 99.09s    |
|                           | 128             | 16        | 30.07              | 1000          | Intel(R) Xeon(R) Silver 4210| DDR4-2933  | 0.5695s       | 32.69s    |
|                           | 128             | 16        | 29.57              | 1000          | AMD Ryzen 5 5600X           | DDR4-3200  | 0.3217s       | 33.50s    |
|                           | 128             | 32        | 11.19              | 100           | Intel(R) Core(TM) i7-7560U  | DDR3-1866  | 0.0846s       | 8.85s     |
|                           | 512             | 16        | 5.92               | 1000          | AMD Ryzen 5 5600X           | DDR4-3200  | 0.3221s       | 168.61s   |
