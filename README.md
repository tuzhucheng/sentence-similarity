# sentence-similarity

I plan to implement some models for sentence similarity found in the literature to reproduce and study them.
They have a wide variety of application, including:

* **Paraphrase Detection**: Give two sentences, are the sentences paraphrases of each other?
* **Semantic Texual Similarity**: Given two sentences, how close are they in terms of semantic equivalence?
* **Natural Language Inference / Textual Entailment**: Can one sentence be inferred from another sentence (the premise)?
* **Answer Selection**: Given question-answer pairs, rank candidate answers based on relevance to question.

## Setup

Install packages in `requirements.txt`.

The`ignite` library, currently in alpha, needs to be installed from source. See https://github.com/pytorch/ignite.

Download SpaCy English model:
```
python -m spacy download en
```

Compile trec_eval for computing MAP/MRR metrics for WikiQA dataset:
```bash
cd metrics
./get_trec_eval.sh
```

## Running

### Baseline

*SICK*
```bash
# Unsupervised
$ python main.py --model sif --dataset sick --unsupervised
Test Results - Epoch: 0 pearson: 0.7199 spearman: 0.5956
# Supervised
$ python main.py --model sif --dataset sick
Test Results - Epoch: 15 pearson: 0.7763 spearman: 0.6637
```

## Attribution

The English Wikipedia token frequency dataset for estimating p(w) in the baseline model is obtained from the official
SIF implementation: https://github.com/PrincetonML/SIF.
