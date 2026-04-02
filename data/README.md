# Data Acquisition

This repository uses video misinformation datasets for evaluation. Due to licensing and author restrictions, some datasets are not included in this repository.

## Datasets

### WildFake

- **Status**: Publicly available
- **Included**: Yes (HMVD_subset_with_transcripts_expanded.jsonl)
- **Source**: [WildFake Dataset](https://github.com/]

### FakeSV

- **Status**: Not included - requires author permission
- **Included**: No
- **How to obtain**: Please contact the original authors of the FakeSV dataset to request access.

### FakeTT

- **Status**: Not included - requires author permission
- **Included**: No
- **How to obtain**: Please contact the original authors of the FakeTT dataset to request access.

## Data Format

All datasets should be in JSONL format with the following fields:

```json
{
  "video_id": "unique_identifier",
  "title": "video title",
  "transcript": "video transcript or audio transcription",
  "annotation": "fake" or "real",
  "fake_type": "type of fake (e.g., 'AI Generation', 'Video Tampering')"
}
```

## Usage

Once you have obtained the datasets, place them in the `data/` directory:

```
data/
├── FakeSV.jsonl
├── FakeTT.jsonl
└── WildFake.jsonl  (or HMVD_subset_with_transcripts_expanded.jsonl)
```

Then run the scripts:

```bash
# Generate comments
python run_conattacker.py --data data/FakeSV.jsonl ...

# Run detection
python run_misguard.py --data data/FakeSV.jsonl ...
```