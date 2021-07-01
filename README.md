# transformer-speed

Author: Dennis Aumiller, Heidelberg University

This repository provides a quick evaluation setup to benchmark wall clock timings of Transformer networks. 
Specifically, this also collects results from various systems.
Feel free to open a PR with timings with your local hardware,
to get a better understanding of how fast/slow inference on various systems is.

## Caveats
Currently, this only supports CPU execution, but I plan to incorporate GPU inference soon.
Note that the latter also suffers from overhead that comes from shuffling memory between the CPU/GPU memories,
which makes the results not directly comparable.
Especially when working with small batch sizes, the overhead of transfer might exceed the performance boost.
Further, existing computing infrastructure supports mainly CPU
(especially in smaller companies that don't rely on Cloud Service provides like AWS/GCP/Azure).


# Results

|Model                      | Sequence Length | CPU                         | RAM        | Tokenization  | Inference | Samples per s |
| :------------------------ | --------------- | :-------------------------- | :--------: | ------------: | --------: | ------------: |
| `bert-base-uncased`       | 64              | Intel(R) Core(TM) i7-7560U  | DDR3-1866  | 0.0581s       | 71.68s    |               |
| `distilbert-base-uncased` | 64              | Intel(R) Core(TM) i7-7560U  | DDR3-1866  | 0.0581s       | 40.77s    |               |
