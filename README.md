# SAFER-Bench

SAFER-Bench is a comprehensive benchmark framework for evaluating Federated Retrieval-Augmented Generation (RAG) systems with approval-based privacy controls. It enables systematic evaluation of retrieval algorithms, merging strategies, LLMs, and privacy mechanisms in realistic federated settings where Data Owners (DOs) maintain control over their data.


### Benchmarking Dimensions
#### Baseline Comparison
- Consider comparing against centralized RAG and simple federated approaches without approval mechanisms to quantify privacy-utility tradeoffs
- Scalability metrics: Test with varying numbers of DOs (2 → 10+) to understand federation overhead
-
#### 🔍 Retrieval Algorithms (DO-side)

##### Vector Search Methods

- **FAISS variants**: IVF, HNSW, LSH, PQ
- **Dense retrievers**: Different embedding models (`PubMedBERT`, `BioBERT`, `Contriever`)
- **Sparse retrievers**: BM25, TF-IDF
- **Hybrid approaches**: Dense + Sparse fusion
- **Approximate vs Exact**: Speed/accuracy tradeoffs

##### Retrieval Strategies

- **Top-k selection**: `k=1, 5, 10, 20`
- **Similarity metrics**: Cosine, L2, Inner Product
- **Chunk strategies**: Size (256, 512, 1024), overlap (0, 50, 100)
- **Re-ranking**: Cross-encoders, ColBERT

#### 🔀 Merging Algorithms (Server-side)

##### Fusion Methods

- **RRF (Reciprocal Rank Fusion)**: Different k parameters
- **Score-based merging**: Weighted average, max, sum
- **Borda count**: Voting-based aggregation
- **CombMNZ**: Score combination with boost
- **Learning-to-rank**: ML-based merging

##### Aggregation Strategies

- **Early fusion**: Merge before retrieval
- **Late fusion**: Merge after retrieval
- **Hierarchical**: Two-stage merging
- **Consensus-based**: Require agreement from multiple DOs

#### 🤖 LLM Components

##### Medical Models

```python
medical_llms = {
    "tiny": ["BioGPT-350M", "ClinicalT5-small"],
    "small": ["BioMistral-7B-Q4", "Meditron-7B-Q4", "Med-PaLM-2B"],
    "general": ["Qwen2.5-0.5B", "SmolLM2-360M", "Llama-3.2-1B"]
}
```

##### Generation Strategies

- **Prompting techniques**: Zero-shot, few-shot, chain-of-thought
- **Context window usage**: 512, 1024, 2048, 4096 tokens
- **Temperature settings**: 0.1, 0.5, 0.7, 1.0
- **Decoding methods**: Greedy, beam search, sampling

#### 🤝 DO Approval Percentages
- Compare metrics when
	- 100% DOs approve the job
	- 75% DOs approve the job
	- 50% DOs approve the job
	- 25% DOs approve the job
- What's the relationship of accuracy and % approval?
#### 🔐 Privacy & Security Mechanisms

##### Approval Mechanisms

- **Static analysis**: Code complexity limits
- **Dynamic sandboxing**: Resource limits
- **Consensus requirements**: 1-of-n, majority, unanimous
- **Temporal policies**: Time-based approval windows

##### Privacy Techniques

- **Differential Privacy**: ε values (0.1, 1.0, 10.0)
- **K-anonymity**: Minimum result set sizes
- **Query limiting**: Rate limiting, budget allocation
- **Output filtering**: Redaction, aggregation

#### 📊 Federation Configurations

##### Network Topologies
- **Centralized**: Star topology with server
- **Hierarchical**: Multi-level aggregation
- **Peer-to-peer**: Direct DO communication
- **Hybrid**: Mixed architectures

##### Data Distributions
- **IID**: Uniform data distribution
- **Non-IID**: Skewed by domain, size, quality
- **Temporal drift**: Changing data over time
- **Missing modalities**: Some DOs lack certain data types

#### 🎯 Evaluation Scenarios

##### Query Types

```python
query_scenarios = {
    "factual": "What is the treatment for diabetes?",
    "reasoning": "Why does X cause Y?",
    "comparative": "Compare treatment A vs B",
    "temporal": "Latest guidelines for...",
    "multi-hop": "Complex reasoning chains"
}
```

##### Adversarial Scenarios

- **Data extraction**: Trying to reveal private info
- **Model inversion**: Reconstructing training data
- **Poisoning**: Malicious responses
- **Free-riding**: Getting results without contributing

#### 📈 Comprehensive Metrics Framework

##### Quality Metrics

```python
quality_metrics = {
    "retrieval": ["precision@k", "recall@k", "MRR", "NDCG"],
    "generation": ["ROUGE", "BLEU", "BERTScore", "exact_match"],
    "medical": ["clinical_accuracy", "guideline_compliance"],
    "end_to_end": ["answer_correctness", "faithfulness", "relevance"]
}
```

##### Efficiency Metrics

```python
efficiency_metrics = {
    "latency": ["retrieval_time", "merge_time", "generation_time"],
    "throughput": ["queries_per_second", "documents_processed"],
    "resource": ["memory_usage", "cpu_utilization", "bandwidth"],
    "scalability": ["performance_vs_num_dos", "performance_vs_data_size"]
}
```

##### Privacy-Utility Metrics

```python
privacy_metrics = {
    "approval": ["approval_rate", "false_positive_rate", "review_time"],
    "leakage": ["mutual_information", "membership_inference_risk"],
    "utility_loss": ["accuracy_drop_with_privacy", "retrieval_degradation"],
    "compliance": ["HIPAA_adherence", "GDPR_compliance"]
}
```

#### 🧪 Experimental Matrix

##### Sample Experimental Design

```python
benchmark_matrix = {
    "baseline": {
        "retriever": "FAISS-IVF",
        "merger": "RRF",
        "llm": "Qwen2.5-0.5B",
        "privacy": "none"
    },
    "dimensions_to_vary": [
        ("retriever", ["FAISS-IVF", "FAISS-HNSW", "BM25", "Hybrid"]),
        ("merger", ["RRF", "Score-avg", "Borda", "CombMNZ"]),
        ("llm", ["Qwen2.5-0.5B", "BioMistral-7B-Q4", "SmolLM2-360M"]),
        ("privacy", ["none", "DP-1.0", "DP-0.1", "k-anon-5"]),
        ("num_data_owners", [2, 3, 5, 10]),
        ("approval_rate", [0.5, 0.7, 0.9, 1.0]),
    ]
}
```
