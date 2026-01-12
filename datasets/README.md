# Datasets

This document describes the dataset structure, configuration, and usage in SAFER-Bench.

## Directory Structure

```
datasets/
├── subsets/                    # Small datasets for rapid prototyping
│   ├── statpearls/
│   │   ├── private/            # Private data (used by DOs)
│   │   └── mock/               # Mock data (visible to DS)
│   ├── textbooks/
│   │   ├── private/
│   │   └── mock/
│   └── mimic-iv-note/
│       ├── private/chunk/      # Clinical note chunks
│       ├── mock/
│       └── specialty_mapping.json  # Chunk-to-specialty mapping
├── full/                       # Complete datasets for production benchmarking
│   ├── statpearls/
│   │   ├── private/
│   │   └── mock/
│   ├── textbooks/
│   │   ├── private/
│   │   └── mock/
│   └── mimic-iv-note/
│       ├── private/chunk/
│       ├── mock/
│       └── specialty_mapping.json  # Generated via: just generate-specialty-mapping
└── mirage_qa.json              # Evaluation questions (shared)
```

## Dataset Modes: Subset vs Full

SaferBench supports two modes controlled by `dataset.use_subset` in `configs/config.yaml`:

### Subset Mode (`use_subset: true`)
- **Purpose**: Fast prototyping, debugging, and testing
- **Path**: `datasets/subsets/{dataset_name}/`
- **Recommended for**: Development, CI/CD, quick iterations

### Full Mode (`use_subset: false`)
- **Purpose**: Production benchmarking and research
- **Path**: `datasets/full/{dataset_name}/`
- **Recommended for**: Publication-ready experiments, final results

## Dataset Statistics

| Dataset | Mode | Chunks | FAISS Index | Doc IDs | Speed | Memory |
|---------|------|--------|-------------|---------|-------|--------|
| **StatPearls** | Subset | ~482 | 26 MB | 1.5 MB | Fast | ~30 MB |
| | Full | ~9,524 | 509 MB | 29 MB | Slower | ~550 MB |
| **Textbooks** | Subset | ~3 | 6.1 MB | 333 KB | Fast | ~10 MB |
| | Full | ~19 | 180 MB | 13 MB | Slower | ~200 MB |

**Size Difference**: Full mode contains ~20x more chunks than subset mode.

## Impact on Benchmarking

| Aspect | Subset Mode | Full Mode |
|--------|-------------|-----------|
| Retrieval Quality | Lower (limited context) | Higher (comprehensive) |
| Statistical Validity | ❌ Not representative | ✅ Representative |
| Execution Time | ~seconds | ~minutes |
| Use Case | Debugging, testing | Research, production |

**Important**:
- Subset mode results should **NOT** be used in research papers
- For publication-ready results, always use full mode with sufficient questions (50-100+)
- The number of evaluation questions is **independent** of subset/full mode (controlled by `qa.num_questions`)

## Clients' Corpuses (Knowledge Bases)

### StatPearls
- **Source**: 9,330 publicly available StatPearls articles
- **Link**: https://www.ncbi.nlm.nih.gov/books/NBK430685/
- **Description**: Peer-reviewed medical reference articles covering clinical topics

### Textbooks
- **Source**: MedQA medical textbooks dataset
- **Reference**: Jin, Di, et al. "What disease does this patient have? a large-scale open domain question answering dataset from medical exams." Applied Sciences 11, no. 14 (2021): 6421.
- **Link**: https://github.com/jind11/MedQA
- **Description**: Medical textbook content used for exam question answering

### MIMIC-IV-Ext-BHC
- **Source**: Labeled Clinical Notes Dataset for Hospital Course Summarization v1.2.0
- **Link**: https://physionet.org/content/labelled-notes-hospital-course/1.2.0/
- **Description**: Clinical notes with hospital course summaries

### MIMIC-IV-Note
- **Source**: Deidentified free-text clinical notes v2.2
- **Link**: https://www.physionet.org/content/mimic-iv-note/2.2/
- **Description**: Clinical documentation from ICU patients

## Server's QA Benchmarks

[MIRAGE QA](https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/refs/heads/main/benchmark.json) benchmark datasets:

- **MedQA**: Multiple-choice medical exam questions from the United States Medical Licensing Examination (USMLE)
- **MedMCQA**: Multiple-choice questions from Indian medical entrance exams (AIIMS & NEET PG)
- **PubMedQA**: Yes/no/maybe questions derived from PubMed biomedical research article abstracts
- **BioASQ**: Biomedical question answering from scientific literature with yes/no and factoid questions
- **MMLU**: Massive Multitask Language Understanding benchmark covering medical topics

## Configuration

### Switching Between Modes

Edit `configs/config.yaml`:

```yaml
dataset:
  use_subset: false  # Set to true for subset, false for full

qa:
  datasets: "pubmedqa|bioasq"  # Which QA datasets to evaluate
  num_questions: 10  # Questions per dataset (set to null for all)
```

### For Research/Publication

```yaml
dataset:
  use_subset: false  # Use full datasets

qa:
  num_questions: 100  # Or null for all questions (~500 per dataset)
```

## Dataset Processing

Datasets are processed according to: https://huggingface.co/datasets/khoaguin/medical-corpus

Each dataset contains:
- **private/**: Actual corpus data (FAISS index, chunks, doc IDs)
- **mock/**: Sample data for testing (subset of private)
- **README.md**: Dataset description

## Specialty-Based Distribution (MIMIC-IV-Note)

The MIMIC-IV-Note dataset supports specialty-based partitioning, where clinical notes are classified into medical specialties for distribution across Data Owners.

### Specialty Mapping

The `specialty_mapping.json` file maps each chunk ID to its classified medical specialty:

```json
{
  "discharge_10005812": "general",
  "discharge_10009614": "gastroenterology",
  "discharge_10009686": "cardiology",
  ...
}
```

### Generating Specialty Mapping

Run the mapping generator before using specialty-based federation:

```bash
just generate-specialty-mapping              # Both subset and full
just generate-specialty-mapping --subset-only
just generate-specialty-mapping --full-only
```

Or directly:
```bash
uv run python scripts/generate_specialty_mapping.py [--subset-only|--full-only]
```

### Medical Specialties

| Specialty | Description | Example Keywords |
|-----------|-------------|------------------|
| Cardiology | Heart and cardiovascular | cardiac, coronary, arrhythmia, troponin |
| Oncology | Cancer and tumors | cancer, chemotherapy, metastasis, carcinoma |
| Neurology | Brain and nervous system | stroke, seizure, Parkinson's, EEG |
| Pulmonology | Lung and respiratory | pneumonia, COPD, ventilator, ARDS |
| Gastroenterology | Digestive system | liver, hepatitis, colonoscopy, cirrhosis |
| Nephrology | Kidney and renal | dialysis, creatinine, CKD, transplant |
| General | Infectious/miscellaneous | sepsis, infection, fever, antibiotic |

### Dataset Statistics (Subset)

| Specialty | Chunks | Percentage |
|-----------|--------|------------|
| Cardiology | ~1,200 | ~16% |
| Oncology | ~800 | ~11% |
| Neurology | ~900 | ~12% |
| Pulmonology | ~1,100 | ~15% |
| Gastroenterology | ~700 | ~10% |
| Nephrology | ~600 | ~8% |
| General | ~2,000 | ~28% |

### Classification Method

Notes are classified using keyword-based matching with section weighting:
- **Discharge Diagnosis**: 10x weight
- **Chief Complaint**: 8x weight
- **Assessment & Plan**: 5x weight
- **History of Present Illness**: 3x weight

See `src/safer_bench/specialty_classifier.py` for implementation details.
