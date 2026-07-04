import os
import mmh3
import nltk
from pathlib import Path
from collections import defaultdict


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
        
