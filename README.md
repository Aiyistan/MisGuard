# MisGuard: Video Misinformation Detection Benchmark

This repository contains the code for the paper:
**"The Echo Chamber Effect: Benchmarking Video Misinformation Detection Under Synthetic Social Consensus"**

## Overview

- **ConAttacker**: Multi-agent simulation framework for generating adversarial comments
- **MisGuard**: Multi-agent reasoning framework for robust video misinformation detection

## Project Structure

```
MisGuard/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.py              # Configuration settings
├── conattacker.py         # ConAttacker implementation
├── misguard.py            # MisGuard implementation
├── dataset.py             # Data loading utilities
├── utils.py               # Utility functions
├── run_conattacker.py     # Generate adversarial comments
├── run_misguard.py        # Run detection
├── data/                  # Data directory
│   └── README.md          # Data acquisition instructions
└── outputs/               # Output directory
    └── sample_comments/   # Sample generated comments
```

## Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/MisGuard
cd MisGuard

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional
export SERPER_API_KEY="your_serper_api_key"  # Optional, for web search
```

Or modify them directly in `config.py`.

## Usage

### Generate Adversarial Comments

Use ConAttacker to generate adversarial comments for videos:

```bash
python run_conattacker.py \
    --data data/sample.jsonl \
    --frames_dir /path/to/frames \
    --output outputs/comments \
    --num_comments 15 \
    --model gpt-4o-mini
```

Arguments:
- `--data`: Path to dataset file (JSONL format)
- `--frames_dir`: Directory containing video frames
- `--output`: Output directory for generated comments
- `--num_comments`: Number of comments to generate per video
- `--model`: LLM model to use
- `--stance`: Comment stance (auto/support/doubt)

### Run Video Misinformation Detection

Use MisGuard to detect video misinformation:

```bash
python run_misguard.py \
    --data data/sample.jsonl \
    --frames_dir /path/to/frames \
    --comments_dir outputs/comments \
    --output outputs/predictions.json \
    --model gpt-4o-mini
```

Arguments:
- `--data`: Path to dataset file
- `--frames_dir`: Directory containing video frames
- `--comments_dir`: Directory containing generated comments
- `--output`: Output file for predictions
- `--model`: LLM model to use
- `--search_api_key`: Serper API key for web search

## Framework Details

### ConAttacker

ConAttacker simulates multiple types of social media users to generate adversarial comments:

- **Witness**: Claims to be at the scene
- **Expert**: Provides technical analysis
- **Knowledge Expert**: Links to external events
- **Skeptic**: Raises doubts and questions
- **Passerby**: Regular user reactions

The framework generates tree-structured comment threads that can mislead video misinformation detectors.

### MisGuard

MisGuard uses a multi-agent reasoning pipeline to detect video misinformation:

1. **Comment Intent Analysis**: Analyzes the intent behind comments
2. **Comment Reflection**: Reflects on whether comments are misleading
3. **Internal Visual Analysis**: Analyzes video frames for inconsistencies
4. **Web Search**: Retrieves external evidence
5. **Final Decision**: Makes a robust prediction

## Benchmark Statistics

- **72,312** comments across 3 datasets
- **10** MLLMs evaluated
- **3** datasets: FakeSV, FakeTT, WildFake

## Citation

If you use this code in your research, please cite:

```bibtex
@article{misguard2024,
  title={The Echo Chamber Effect: Benchmarking Video Misinformation Detection Under Synthetic Social Consensus},
  author={Anonymous Authors},
  journal={Anonymous Venue},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.