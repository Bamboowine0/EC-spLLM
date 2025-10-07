# EC-spLLM

A  evolutionary framework for KG-augmented LLMs that instills epistemic cognition into the LLM to systematically exploit both its internal knowledge and expert knowledge.

## Quick Start

### Prerequisites

```bash
pip install openai chromadb sentence-transformers bert-score tqdm numpy
```

### Configuration

Configure your OpenAI API key in `run_pipeline.py`:

```python
opeani_api_keys = {
    "your_key": "sk-xxxx"
}
```

### Run

```bash
python run_pipeline.py --dataset pubmed_qa --test_num 100 --depth 3 --width 5
```

### Key Parameters

- `--dataset`: Dataset name
- `--test_num`: Number of test questions
- `--depth`: Retrieval depth (default: 3)
- `--width`: Retrieval width per layer (default: 5)
- `--sim_threshold`: Similarity threshold (default: 0.4)
- `--model`: LLM model to use (default: gpt-4.1)

## Supported Datasets

- **PubMed QA**: Medical question answering
- **BioASQ**: Biomedical question answering
- **LawBenchmark**: Legal question answering
- **NQ-open**: Natural questions open domain
- **SimpleQA**: Science question answering

## Notes

- Ensure sufficient OpenAI API quota
- SentenceTransformer model will be downloaded on first run
- GPU recommended for better performance
- Knowledge graph grows dynamically during QA process
- Results are saved in JSONL format with detailed metrics
