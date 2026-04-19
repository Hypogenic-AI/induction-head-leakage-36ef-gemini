# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-2

### Overview
- **Source**: HuggingFace (wikitext-2-raw-v1)
- **Size**: ~4.3K samples (test split)
- **Format**: HuggingFace Dataset
- **Task**: Language Modeling Evaluation
- **Splits**: test (4358 examples)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset.save_to_disk("datasets/wikitext-2")
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-2")
```

## Dataset 2: Synthetic Markov Chain ICL (to be generated)

### Overview
- **Source**: Generated using `code/Evolution-of-Statistical-Induction-Heads/`
- **Size**: Configurable (e.g., 10K samples)
- **Format**: JSON/Tensors
- **Task**: In-context Learning of Markov Chains
- **Notes**: Essential for studying how transformers learn statistics and form induction heads.

### Generation Instructions

See `code/Evolution-of-Statistical-Induction-Heads/README.md` for generation scripts.
