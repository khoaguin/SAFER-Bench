# Metrics Implementation Plan for SaferBench

## Overview

This document outlines the implementation plan for adding comprehensive evaluation metrics to SaferBench, as defined in `configs/config.yaml` (lines 29-61).

## Current Status

### ✅ Already Implemented

The following metrics are currently implemented and working:

- **Basic Accuracy Metrics**
  - Overall weighted accuracy across datasets
  - Per-dataset accuracy

- **Basic Timing Metrics**
  - Mean query time per dataset
  - Total benchmark duration

- **Execution Metrics**
  - Job success/failure rates
  - DS server status

- **Federation Configuration**
  - Federation size (number of DOs)
  - Data owner details (email, dataset, data fraction)

### ❌ Not Yet Implemented

The following metrics from `config.yaml` need to be implemented:

## Metric Categories to Implement

### 1. Quality Metrics (Accuracy Breakdown)

**Config Reference:** `evaluation.accuracy_metrics` (lines 32-37)

- [ ] **per_option_accuracy** - Accuracy breakdown by multiple-choice option (A/B/C/D)
  - Calculate accuracy for each option separately
  - Useful for detecting model bias towards certain options

- [ ] **confusion_matrix** - Expected vs Predicted answer distribution
  - 4x4 matrix for multiple-choice questions (A/B/C/D)
  - Shows which answers are commonly confused

- [ ] **option_distribution** - Which options the model selects most often
  - Histogram of predicted answer frequencies
  - Detects if model has option bias (e.g., always choosing A)

### 2. Efficiency Metrics (Timing Breakdown)

**Config Reference:** `evaluation.efficiency_metrics` (lines 40-45)

- [ ] **retrieval_time** - Federated document retrieval time (seconds)
  - Time from query submission to receiving all documents from DOs

- [ ] **merge_time** - Document merging (RRF) time (seconds)
  - Time to apply Reciprocal Rank Fusion and merge documents

- [ ] **generation_time** - LLM inference time (seconds)
  - Time for LLM to generate the answer

- [ ] **total_query_time** - End-to-end query time (seconds)
  - Total time from question submission to final answer

- [ ] **queries_per_second** - Throughput (QPS)
  - Number of questions processed per second

### 3. Privacy-Utility Tradeoff Metrics

**Config Reference:** `evaluation.privacy_metrics` (lines 48-50)

- [x] **approval_rate** - Percentage of DOs that approved jobs
  - Already captured in `JobProcessingResult.approval_rate`
  - Needs to be extracted and added to final metrics

- [ ] **accuracy_drop_with_privacy** - Accuracy loss compared to 100% approval baseline
  - Requires baseline run with 100% approval
  - Compare accuracy with current approval rate vs baseline
  - Formula: `baseline_accuracy - current_accuracy`

### 4. Federation-Specific Metrics

**Config Reference:** `evaluation.federation_metrics` (lines 53-60)

- [x] **federation_size** - Number of participating DOs
  - Already implemented

- [ ] **participation_rate** - % of DOs that responded to queries
  - Track which DOs responded vs total DOs
  - Formula: `responding_dos / total_dos`

- [ ] **communication_time** - Network time for federated retrieval (seconds)
  - Time spent in network communication during retrieval
  - Can be approximated as `retrieval_time - local_processing_time`

- [ ] **communication_ratio** - Network time / total query time
  - Formula: `communication_time / total_query_time`

- [ ] **per_do_contribution** - % of final documents from each DO
  - Track which DO contributed which documents
  - Calculate percentage contribution after RRF merging

- [ ] **data_source_diversity** - Number of unique sources in merged docs
  - Count distinct DOs in final top-k documents
  - Higher = better data diversity

- [ ] **merge_strategy_impact** - Comparison of merging strategies (future)
  - Compare RRF vs other merging strategies
  - Marked as future work

## Implementation Strategy

### Phase 1: Enhance FedRAG Output (Tasks 1-3)

Modify `src/safer_bench/assets/fedrag_v1/fedrag/server_app.py` to capture and output detailed metrics.

#### Task 1: Output Detailed Timing Metrics

**Location:** `server_app.py` lines 148-210

**Changes:**
```python
# Current: Only tracks total question time
q_st = time.time()
docs, scores = submit_question(...)  # retrieval
merged_docs = merge_documents(...)   # merge
prompt, predicted_answer = llm_querier.answer(...)  # generation
q_et = time.time()
q_time = q_et - q_st

# Enhanced: Track each stage separately
q_st = time.time()
retrieval_st = time.time()
docs, scores = submit_question(...)
retrieval_time = time.time() - retrieval_st

merge_st = time.time()
merged_docs = merge_documents(...)
merge_time = time.time() - merge_st

gen_st = time.time()
prompt, predicted_answer = llm_querier.answer(...)
generation_time = time.time() - gen_st
q_time = time.time() - q_st

# Store timing breakdown
timing_breakdown[dataset_name].append({
    'retrieval': retrieval_time,
    'merge': merge_time,
    'generation': generation_time,
    'total': q_time
})
```

**Output Format:**
```
Timing Breakdown: {dataset_name}
Mean Retrieval Time: 0.45s
Mean Merge Time: 0.12s
Mean Generation Time: 2.31s
Mean Total Query Time: 2.88s
Queries Per Second: 0.35
```

#### Task 2: Output Accuracy Breakdown

**Location:** `server_app.py` lines 195-210

**Changes:**
```python
# Track predictions by option
option_predictions[dataset_name] = defaultdict(int)
option_correct[dataset_name] = defaultdict(int)
confusion_matrix[dataset_name] = defaultdict(lambda: defaultdict(int))

# In the evaluation loop
for dataset_name in qa_datasets:
    for q in datasets[dataset_name]:
        # ... existing code ...
        if predicted_answer is not None:
            # Track predictions
            option_predictions[dataset_name][predicted_answer] += 1
            if predicted_answer == answer:
                option_correct[dataset_name][predicted_answer] += 1
            confusion_matrix[dataset_name][answer][predicted_answer] += 1
```

**Output Format:**
```
Accuracy Breakdown: {dataset_name}
Per-Option Accuracy:
  Option A: 0.75 (15/20)
  Option B: 0.60 (12/20)
  Option C: 0.80 (16/20)
  Option D: 0.70 (14/20)

Option Distribution (Predicted):
  A: 22%, B: 28%, C: 25%, D: 25%

Confusion Matrix:
     A    B    C    D
A   15    3    1    1
B    2   12    4    2
C    1    2   16    1
D    2    3    2   14
```

#### Task 3: Output Federation Metrics

**Location:** `server_app.py` `submit_question()` function

**Changes:**
```python
def submit_question(...):
    # Track which nodes contributed documents
    node_contributions = {}

    for reply in replies:
        if reply.has_content():
            node_id = reply.metadata.get('src_node_id')
            corpus_name = reply.content.get('corpus_name')
            num_docs = len(reply.content["docs_n_scores"]["documents"])
            node_contributions[node_id] = {
                'corpus': corpus_name,
                'num_docs': num_docs
            }

    return documents, scores, node_contributions

# After merging, track which DOs contributed to final top-k
def track_do_contribution(merged_docs, all_docs, node_contributions):
    do_contribution = defaultdict(int)
    for merged_doc in merged_docs:
        # Find which DO contributed this document
        for node_id, contrib in node_contributions.items():
            if merged_doc in contrib['docs']:
                do_contribution[contrib['corpus']] += 1
    return do_contribution
```

**Output Format:**
```
Federation Metrics:
Participation Rate: 100% (2/2 DOs responded)
Data Source Diversity: 2 unique sources
Per-DO Contribution (top-8 docs):
  statpearls: 62.5% (5/8 docs)
  textbooks: 37.5% (3/8 docs)
```

### Phase 2: Update Data Models (Tasks 4-6)

Update Pydantic models in `src/safer_bench/models.py` to support new metrics.

#### Task 4: Update TimingBreakdown Model

**Location:** `models.py` lines 198-222

**Changes:**
```python
class TimingBreakdown(BaseModel):
    """Detailed timing breakdown for efficiency analysis."""

    retrieval_time: Optional[float] = Field(
        None, ge=0.0, description="Federated document retrieval time (seconds)"
    )
    merge_time: Optional[float] = Field(
        None, ge=0.0, description="Document merging (RRF) time (seconds)"
    )
    generation_time: Optional[float] = Field(
        None, ge=0.0, description="LLM inference time (seconds)"
    )
    total_query_time: Optional[float] = Field(
        None, ge=0.0, description="End-to-end query time (seconds)"
    )
    queries_per_second: Optional[float] = Field(
        None, ge=0.0, description="Throughput (QPS)"
    )
    communication_time: Optional[float] = Field(
        None, ge=0.0, description="Network time for federated retrieval (seconds)"
    )
    communication_ratio: Optional[Percentage] = Field(
        None, description="Network time / total query time"
    )
```

#### Task 5: Create Accuracy Breakdown Models

**Location:** `models.py` (new models after DatasetMetrics)

**New Models:**
```python
class PerOptionAccuracy(BaseModel):
    """Accuracy breakdown by multiple-choice option."""

    option: str = Field(..., description="Option label (A/B/C/D)")
    accuracy: Percentage = Field(..., description="Accuracy for this option")
    correct: int = Field(..., ge=0, description="Number of correct predictions")
    total: int = Field(..., ge=0, description="Total questions with this answer")


class OptionDistribution(BaseModel):
    """Distribution of predicted options."""

    option_counts: Dict[str, int] = Field(..., description="Count per option")
    option_percentages: Dict[str, Percentage] = Field(
        ..., description="Percentage per option"
    )


class ConfusionMatrix(BaseModel):
    """Confusion matrix for multiple-choice QA."""

    matrix: Dict[str, Dict[str, int]] = Field(
        ..., description="Expected (row) vs Predicted (col) counts"
    )


class AccuracyBreakdown(BaseModel):
    """Complete accuracy analysis for a dataset."""

    overall_accuracy: Percentage
    per_option: List[PerOptionAccuracy]
    option_distribution: OptionDistribution
    confusion_matrix: ConfusionMatrix
```

#### Task 6: Create Federation Metrics Models

**Location:** `models.py` (new models)

**New Models:**
```python
class DataOwnerContribution(BaseModel):
    """Contribution of a single data owner to final results."""

    do_email: EmailStr
    dataset: str
    num_documents: int = Field(..., ge=0, description="Docs in final top-k")
    contribution_percentage: Percentage = Field(
        ..., description="% of final documents"
    )


class FederationPerformanceMetrics(BaseModel):
    """Federation-specific performance metrics."""

    participation_rate: Percentage = Field(
        ..., description="% of DOs that responded to queries"
    )
    data_source_diversity: int = Field(
        ..., ge=0, description="Number of unique sources in merged docs"
    )
    per_do_contribution: List[DataOwnerContribution] = Field(
        ..., description="Contribution breakdown per DO"
    )
    communication_time: Optional[float] = Field(
        None, ge=0.0, description="Network communication time (seconds)"
    )
    communication_ratio: Optional[Percentage] = Field(
        None, description="Network time / total time ratio"
    )


class PrivacyMetrics(BaseModel):
    """Privacy-utility tradeoff metrics."""

    approval_rate: Percentage = Field(..., description="% of DOs that approved")
    target_approval_rate: Percentage = Field(
        ..., description="Configured approval rate"
    )
    accuracy_drop_with_privacy: Optional[float] = Field(
        None, description="Accuracy loss vs 100% baseline"
    )
    baseline_accuracy: Optional[Percentage] = Field(
        None, description="Accuracy with 100% approval"
    )
```

### Phase 3: Enhance Metrics Collection (Tasks 7-10)

Update `src/safer_bench/metrics_collector.py` to extract new metrics.

#### Task 7: Extract Timing Metrics

**Location:** `metrics_collector.py` `parse_ds_stdout()` method

**Changes:**
```python
def parse_ds_stdout(self, stdout: str) -> Dict[str, Any]:
    """Parse DS server stdout to extract all metrics."""

    # Existing dataset metrics pattern (accuracy, questions)
    # ... keep existing code ...

    # NEW: Parse timing breakdown
    timing_pattern = r"Timing Breakdown: (\w+)\s*\n" \
                     r".*?Mean Retrieval Time: ([\d.]+)s\s*\n" \
                     r".*?Mean Merge Time: ([\d.]+)s\s*\n" \
                     r".*?Mean Generation Time: ([\d.]+)s\s*\n" \
                     r".*?Mean Total Query Time: ([\d.]+)s\s*\n" \
                     r".*?Queries Per Second: ([\d.]+)"

    timing_matches = re.finditer(timing_pattern, stdout, re.MULTILINE | re.DOTALL)

    for match in timing_matches:
        dataset_name = match.group(1)
        if dataset_name in metrics:
            metrics[dataset_name]['timing'] = {
                'retrieval_time': float(match.group(2)),
                'merge_time': float(match.group(3)),
                'generation_time': float(match.group(4)),
                'total_query_time': float(match.group(5)),
                'queries_per_second': float(match.group(6))
            }

    return metrics
```

#### Task 8: Extract Accuracy Breakdown

**Location:** `metrics_collector.py` (new method)

**New Method:**
```python
def parse_accuracy_breakdown(self, stdout: str) -> Dict[str, Dict]:
    """Parse accuracy breakdown from DS stdout."""

    accuracy_breakdown = {}

    # Parse per-option accuracy
    option_pattern = r"Per-Option Accuracy:.*?\n" \
                     r"  Option A: ([\d.]+) \((\d+)/(\d+)\)\s*\n" \
                     r"  Option B: ([\d.]+) \((\d+)/(\d+)\)\s*\n" \
                     r"  Option C: ([\d.]+) \((\d+)/(\d+)\)\s*\n" \
                     r"  Option D: ([\d.]+) \((\d+)/(\d+)\)"

    # Parse option distribution
    dist_pattern = r"Option Distribution \(Predicted\):.*?\n" \
                   r"  A: ([\d.]+)%, B: ([\d.]+)%, C: ([\d.]+)%, D: ([\d.]+)%"

    # Parse confusion matrix
    matrix_pattern = r"Confusion Matrix:.*?\n" \
                     r".*?A\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\n" \
                     r".*?B\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\n" \
                     r".*?C\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\n" \
                     r".*?D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"

    # ... implement parsing logic ...

    return accuracy_breakdown
```

#### Task 9: Extract Federation Metrics

**Location:** `metrics_collector.py` (new method)

**New Method:**
```python
def parse_federation_metrics(self, stdout: str) -> Dict[str, Any]:
    """Parse federation-specific metrics from DS stdout."""

    fed_metrics = {}

    # Parse participation rate
    participation_pattern = r"Participation Rate: ([\d.]+)% \((\d+)/(\d+) DOs responded\)"

    # Parse data source diversity
    diversity_pattern = r"Data Source Diversity: (\d+) unique sources"

    # Parse per-DO contribution
    contribution_pattern = r"Per-DO Contribution.*?:\s*\n" \
                          r"((?:  \w+: [\d.]+% \(\d+/\d+ docs\)\s*\n?)+)"

    # ... implement parsing logic ...

    return fed_metrics
```

#### Task 10: Add Privacy Metrics Calculation

**Location:** `metrics_collector.py` `collect()` method

**Changes:**
```python
async def collect(
    self,
    federation_info: FederationInfo,
    fedrag_results: FedRAGExecutionResult,
    job_processing_results: JobProcessingResult,  # NEW parameter
    start_time: datetime,
    end_time: datetime,
) -> BenchmarkMetrics:
    """Collect all metrics from the benchmark execution."""

    # ... existing code ...

    # NEW: Calculate privacy metrics
    privacy_metrics = PrivacyMetrics(
        approval_rate=job_processing_results.approval_rate,
        target_approval_rate=self.cfg.federation.approval.percentage,
        accuracy_drop_with_privacy=None,  # Requires baseline run
        baseline_accuracy=None
    )

    # If baseline exists, calculate accuracy drop
    baseline_path = self.cfg.get('baseline_metrics_path')
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_acc = baseline['results']['overall']['weighted_accuracy']
        current_acc = overall_metrics.weighted_accuracy
        privacy_metrics.accuracy_drop_with_privacy = baseline_acc - current_acc
        privacy_metrics.baseline_accuracy = baseline_acc

    # ... build BenchmarkMetrics with new fields ...
```

### Phase 4: Integration & Reporting (Tasks 11-14)

#### Task 11: Update BenchmarkMetrics Model

**Location:** `models.py` lines 237-244

**Changes:**
```python
class BenchmarkMetrics(BaseModel):
    """Complete benchmark metrics and results."""

    benchmark_metadata: BenchmarkMetadata
    federation: FederationMetrics
    results: ResultsMetrics
    execution: ExecutionMetrics

    # NEW FIELDS
    privacy: Optional[PrivacyMetrics] = Field(None, description="Privacy metrics")
    federation_performance: Optional[FederationPerformanceMetrics] = Field(
        None, description="Federation-specific performance metrics"
    )
```

Update `ResultsMetrics` to include accuracy breakdown:
```python
class ResultsMetrics(BaseModel):
    """Results metrics containing per-dataset and overall metrics."""

    per_dataset: Dict[str, DatasetMetrics]
    overall: OverallMetrics

    # NEW FIELDS
    accuracy_breakdown: Optional[Dict[str, AccuracyBreakdown]] = Field(
        None, description="Detailed accuracy analysis per dataset"
    )
```

#### Task 12: Update Report Generation

**Location:** `benchmark_runner.py` `_generate_summary()` method

**Changes:**
```python
def _generate_summary(self, metrics: BenchmarkMetrics) -> str:
    """Generate a markdown summary of benchmark results."""

    # ... existing summary code ...

    summary += "\n### Efficiency Metrics\n"
    if metrics.execution.timing:
        timing = metrics.execution.timing
        summary += f"- **Mean Retrieval Time**: {timing.retrieval_time:.3f}s\n"
        summary += f"- **Mean Merge Time**: {timing.merge_time:.3f}s\n"
        summary += f"- **Mean Generation Time**: {timing.generation_time:.3f}s\n"
        summary += f"- **Queries Per Second**: {timing.queries_per_second:.2f}\n"

    summary += "\n### Privacy-Utility Tradeoff\n"
    if metrics.privacy:
        summary += f"- **Approval Rate**: {metrics.privacy.approval_rate:.1%}\n"
        if metrics.privacy.accuracy_drop_with_privacy is not None:
            summary += f"- **Accuracy Drop**: {metrics.privacy.accuracy_drop_with_privacy:.2%}\n"

    summary += "\n### Federation Performance\n"
    if metrics.federation_performance:
        fp = metrics.federation_performance
        summary += f"- **Participation Rate**: {fp.participation_rate:.1%}\n"
        summary += f"- **Data Source Diversity**: {fp.data_source_diversity} sources\n"
        summary += "\n**Per-DO Contribution:**\n"
        for contrib in fp.per_do_contribution:
            summary += f"- {contrib.dataset}: {contrib.contribution_percentage:.1%}\n"

    # Add accuracy breakdown tables if available
    if metrics.results.accuracy_breakdown:
        summary += "\n### Detailed Accuracy Analysis\n"
        for dataset_name, breakdown in metrics.results.accuracy_breakdown.items():
            summary += f"\n#### {dataset_name}\n"
            summary += "| Option | Accuracy | Correct/Total |\n"
            summary += "|--------|----------|---------------|\n"
            for opt_acc in breakdown.per_option:
                summary += f"| {opt_acc.option} | {opt_acc.accuracy:.2%} | "
                summary += f"{opt_acc.correct}/{opt_acc.total} |\n"

    return summary
```

#### Task 13: Add Configuration Validation

**Location:** New file `src/safer_bench/config_validator.py`

**New Module:**
```python
"""Configuration validator for metrics."""

from typing import Set
from omegaconf import DictConfig
from loguru import logger


IMPLEMENTED_ACCURACY_METRICS = {
    'overall_accuracy',
    'per_dataset_accuracy',
    'per_option_accuracy',
    'confusion_matrix',
    'option_distribution'
}

IMPLEMENTED_EFFICIENCY_METRICS = {
    'retrieval_time',
    'merge_time',
    'generation_time',
    'total_query_time',
    'queries_per_second'
}

IMPLEMENTED_PRIVACY_METRICS = {
    'approval_rate',
    'accuracy_drop_with_privacy'
}

IMPLEMENTED_FEDERATION_METRICS = {
    'federation_size',
    'participation_rate',
    'communication_time',
    'communication_ratio',
    'per_do_contribution',
    'data_source_diversity'
}


def validate_metrics_config(cfg: DictConfig) -> bool:
    """Validate that configured metrics are implemented."""

    all_implemented = (
        IMPLEMENTED_ACCURACY_METRICS |
        IMPLEMENTED_EFFICIENCY_METRICS |
        IMPLEMENTED_PRIVACY_METRICS |
        IMPLEMENTED_FEDERATION_METRICS
    )

    issues = []

    # Check accuracy metrics
    for metric in cfg.evaluation.accuracy_metrics:
        if metric not in IMPLEMENTED_ACCURACY_METRICS:
            issues.append(f"Accuracy metric not implemented: {metric}")

    # Check efficiency metrics
    for metric in cfg.evaluation.efficiency_metrics:
        if metric not in IMPLEMENTED_EFFICIENCY_METRICS:
            issues.append(f"Efficiency metric not implemented: {metric}")

    # Check privacy metrics
    for metric in cfg.evaluation.privacy_metrics:
        if metric not in IMPLEMENTED_PRIVACY_METRICS:
            issues.append(f"Privacy metric not implemented: {metric}")

    # Check federation metrics
    for metric in cfg.evaluation.federation_metrics:
        if metric not in IMPLEMENTED_FEDERATION_METRICS:
            issues.append(f"Federation metric not implemented: {metric}")

    if issues:
        logger.warning("Metrics configuration validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    logger.success("✅ All configured metrics are implemented")
    return True
```

Call this validator in `benchmark_runner.py.__init__()`:
```python
from safer_bench.config_validator import validate_metrics_config

def __init__(self, cfg: DictConfig):
    self.cfg = cfg
    validate_metrics_config(cfg)
    # ... rest of init ...
```

#### Task 14: End-to-End Testing

**Testing Plan:**

1. **Unit Tests** - Test individual parsing functions
   ```python
   def test_parse_timing_metrics():
       sample_output = """
       Timing Breakdown: pubmedqa
       Mean Retrieval Time: 0.45s
       Mean Merge Time: 0.12s
       ...
       """
       metrics = MetricsCollector(cfg)
       result = metrics.parse_ds_stdout(sample_output)
       assert result['pubmedqa']['timing']['retrieval_time'] == 0.45
   ```

2. **Integration Tests** - Run full benchmark with small dataset
   ```bash
   safer-bench federation=local_2do qa.num_questions=2
   ```

3. **Validation** - Check generated metrics file
   ```bash
   cat outputs/*/metrics_*.json | jq .
   ```

4. **Report Review** - Verify markdown summary contains all metrics
   ```bash
   cat outputs/*/summary_*.md
   ```

## Implementation Order

### Sprint 1: Timing & Basic Metrics
- Tasks 1, 4, 7 - Timing metrics end-to-end
- Verify timing data flows correctly

### Sprint 2: Accuracy Breakdown
- Tasks 2, 5, 8 - Accuracy breakdown end-to-end
- Verify confusion matrices and distributions

### Sprint 3: Federation & Privacy
- Tasks 3, 6, 9, 10 - Federation and privacy metrics
- Verify per-DO contributions tracked correctly

### Sprint 4: Integration & Polish
- Tasks 11, 12, 13, 14 - Integration, reporting, validation, testing
- Complete end-to-end testing

## Dependencies & Considerations

### External Dependencies
- No new Python packages required
- All metrics use existing standard library or numpy/sklearn

### Configuration Changes
- No changes needed to `config.yaml`
- May add `baseline_metrics_path` for privacy metrics comparison

### Backward Compatibility
- All new metrics are optional fields
- Existing metrics pipeline continues to work
- New fields populated progressively as implemented

### Performance Impact
- Minimal: Most metrics are calculated from data already being collected
- Timing breakdown adds ~3 time.time() calls per question (negligible)
- Confusion matrix tracking adds small memory overhead (4x4 matrix per dataset)

## Success Criteria

Implementation is complete when:

1. ✅ All metrics from `config.yaml` are extracted and stored
2. ✅ Pydantic models validate all metric structures
3. ✅ JSON output contains all metric fields
4. ✅ Markdown report displays all metrics in organized sections
5. ✅ Configuration validator confirms all metrics implemented
6. ✅ Unit tests pass for all parsing functions
7. ✅ Integration test runs successfully end-to-end
8. ✅ Documentation updated with new metrics

## References

- **Config:** `configs/config.yaml` lines 29-61
- **Models:** `src/safer_bench/models.py`
- **Collector:** `src/safer_bench/metrics_collector.py`
- **Runner:** `src/safer_bench/benchmark_runner.py`
- **FedRAG:** `src/safer_bench/assets/fedrag_v1/fedrag/server_app.py`

## Notes

- `merge_strategy_impact` marked as future work (requires multiple merger implementations)
- `accuracy_drop_with_privacy` requires baseline run with 100% approval
- Some federation metrics (per-DO contribution, diversity) require tracking document sources through the pipeline
