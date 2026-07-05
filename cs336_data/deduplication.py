import os
import mmh3
import nltk
from pathlib import Path
from collections import defaultdict


def minhash(S: set[str], seed: int):
    return min(mmh3.hash(s, seed) for s in S)


def generate_ngrams_from_file(ngrams, path):
    with open(path) as f:
        data = f.read()
    tokenzied_data = nltk.word_tokenize(data)
    ngrams_data = nltk.ngrams(tokenzied_data, n=ngrams)
    ngrams_data = [' '.join(data) for data in ngrams_data]
    return ngrams_data


def find_true_jaccard_similarity(candidate1, candidate2):
    ngram1 = generate_ngrams_from_file(candidate1)
    ngram2 = generate_ngrams_from_file(candidate2)
    return 1 - nltk.metrics.jaccard_distance(ngram1, ngram2)

def deduplicate_minhash(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    candidates_dict = defaultdict(list)
    for path in input_files:
        ngrams_data = generate_ngrams_from_file(ngrams, path)
        sign = [minhash(ngrams_data, i) for i in range(num_hashes)]

        num_hash_per_band = num_hashes//num_bands
        for b in range(num_bands):
            key = sign[b*num_hash_per_band:(b+1)*num_hash_per_band]
            candidates_dict[key].append(path)
    
    for candidates in candidates_dict.values():
        num_candidates = len(candidates)
        if num_candidates > 1:
            for i in range(num_candidates-1):
                for j in range(i, num_candidates):
                    sim = find_true_jaccard_similarity(candidates[i], candidates[j])
                    if sim > jaccard_threshold:
                        pass


def deduplicate_exact(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    exact_counts = defaultdict(int)
    for path in input_files:
        with open(path) as f:
            for line in f.readlines():
                hash = mmh3.hash(line)
                exact_counts[hash] += 1
    
    for path in input_files:
        output = []
        with open(path) as f:
            for line in f.readlines():
                hash = mmh3.hash(line)
                count = exact_counts[hash]
                if count > 1:
                    continue
                output.append(line)
        
        output_path = os.path.join(output_directory, Path(path).name)
        with open(output_path, 'w') as f:
            f.writelines(output)
        
