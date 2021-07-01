"""
This script is a minimal example of how quickly (or slowly) Transformer networks can be run for inference.
"""

import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="bert-base-uncased",
                        help="The name of the Transformer model to use for this run.")
    parser.add_argument("max_length", type=int, default=512,
                        help="Maximum input length. Most transformer models restrict this to 512 subword tokens, "
                             "however, they also benefit from a shorter input length due to less calculations.")
    parser.add_argument("--device", default="CPU",
                        help="Which device to use for execution. Currently only supports CPU usage.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="How many samples will be aggregated into a single forward pass. Does not affect the "
                             "total number of --num_samples being processed")
    parser.add_argument("--seed", type=int, default=4321,
                        help="Random seed used by the numpy rng state to determine the samples")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples that should be processed during the entire process. "
                             "Our data set currently provides ~200 inputs (of maximum length), "
                             "which will be chosen randomly, and potentially repeated.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open("inputs.txt") as f:
        sentences = f.readlines()
    # Strip newlines from inputs.
    sentences = [sentence.strip(" \n") for sentence in sentences]
    rng = np.random.default_rng(seed=args.seed)
    samples = rng.choice(sentences, args.num_samples)

    batches = []
    for idx in range(0, args.num_samples, args.batch_size):
        batches.append(list(samples[idx:idx+args.batch_size]))

    # Model loading will not be considered
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    # Tokenization process
    tokenized_batches = []
    start_time_tokenization = time.time()
    for batch in batches:
        tokenized_batches.append(tokenizer(batch, padding=True, truncation=True,
                                           max_length=args.max_length, return_tensors="pt"))
    stop_time_tokenization = time.time()

    start_time_forward_pass = time.time()
    with torch.no_grad():
        for batch in tqdm(tokenized_batches):
            res = model(**batch)
    stop_time_forward_pass = time.time()

    tokenization_duration = stop_time_tokenization - start_time_tokenization
    forward_pass_duration = stop_time_forward_pass - start_time_forward_pass

    print(f"Tokenization took: {tokenization_duration:.4f}s")
    print(f"Forward pass took: {forward_pass_duration:.2f}s")

    print(f"This equals a processing of approximately "
          f"{args.num_samples / (tokenization_duration + forward_pass_duration):.2f} samples per second")
