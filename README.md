# SQL Query Generation from Natural Language

Author: [Yufan Zhang](https://yufanbruce.com)

## Introduction

The project is about generating SQL queries from natural language questions. The dataset used in this project is the Flights dataset, which is a dataset of flights and airports. The dataset contains 25 tables, including airports, airlines, flights, etc. The goal of this project is to generate SQL queries from natural language questions in three approaches:

1. Fine-tuned a pre-trained [T5 model](https://huggingface.co/docs/transformers/en/model_doc/t5) on the Flights dataset.
2. Train a [T5 model](https://huggingface.co/docs/transformers/en/model_doc/t5) from scratch on the Flights dataset.
3. Prompting & In-context Learning with [Gemma 1.1 2B](https://huggingface.co/google/gemma-1.1-2b-it) model.

## Project Structure

The project structure is as follows:

```
.
├── README.md
├── .env  # Environment file to store your HuggingFace Access Token
├── checkpoints
├── data
│   ├── alignment.txt
│   ├── dev.nl
│   ├── dev.sql
│   ├── flight_database.db
│   ├── flight_database.schema
│   ├── test.nl
│   ├── train.nl
│   └── train.sql
├── dataset
│   ├── __init__.py
│   └── sql_dataset.py
├── options
│   ├── __init__.py
│   ├── prompting_options.py
│   └── t5_options.py
├── prompting.py
├── requirements.txt
├── results
│   ├── records
│   └── queries
├── t5.py
└── utils
    ├── __init__.py
    ├── args.py
    ├── data.py
    ├── evaluation.py
    ├── prompting_utils.py
    └── t5_utils.py
```

## How to Run

### Development Environment Setup

To set up the development environment, you can run the following command:

```bash
conda create -n text2sql python=3.10
conda activate text2sql
pip install -r requirements.txt
````

### Fine-tuned T5 Model

To run the fine-tuned T5 model, you can run the following command:

```bash
python3 t5.py --finetune
```

### Train T5 Model from Scratch

To train the T5 model from scratch, you can run the following command:

```bash
python3 t5.py
```

### Prompting & In-context Learning with Gemma 1.1 2B Model

To run the prompting & in-context learning with Gemma 1.1 2B model, you can run the following command:

```bash
python3 prompting.py
```

## Acknowledgement

This project is a part of the course project for the course CS 5740: Natural Language Processing (2024 Spring) at Cornell Tech. 