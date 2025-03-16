## Overview

Reproducibility is an important aspect of research. Experiments are valid if and only if they are reproducible. Incredible state-of-the-art research is released everyday, but a lack of scientific rigor means that many components of machine learning projects are either underspecified or do not exist at all. This slows down progress, making it hard to "stand atop the shoulders of giants" - the spirit of science.

We ❤️ Open Source Science, and want to encourage more researchers to follow this! This project takes on an independent execution of this effort.

## Project Goals

1. Read, understand and ideally perform a blind-paper replication of the original research
   - Create experiments described by the research paper by looking only at the paper, not any provided implementation
   - When information is incomplete or not present in the paper/appendix, either reference the source code or make an educated guess

2. Add ablation studies, comparing the results of your reimplementation with that of the original

3. Compile efforts and learnings into a reproducibility report

## Project Structure

We'll use:
- Git & DVC for hosting and versioning code, data, models, and artifacts
- MLflow for tracking experiments
- Efficient management of unstructured data
- Data pipelines to improve collaboration and make projects easier to understand and reproduce

## Todo List

* [ ] Dataset Preparation
* [ ] VLM Architecture
* [ ] Evaluation
* [ ] DVC Setup
* [ ] Experiment logging with MLflow

[ReScience](https://rescience.github.io/)