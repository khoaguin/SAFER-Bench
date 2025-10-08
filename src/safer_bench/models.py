from typing_extensions import Optional, List, Any, Dict
from enum import Enum

from pydantic import BaseModel, Field, EmailStr, field_validator

from syft_rds.client.rds_client import RDSClient
from syft_rds.models import Job


class DataOwnerInfo(BaseModel):
    """Information about a data owner in the federation."""

    email: EmailStr
    dataset: str = Field(..., min_length=1, description="Dataset name")
    data_fraction: float = Field(
        ..., gt=0.0, le=1.0, description="Fraction of dataset (0-1]"
    )

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Ensure dataset name is valid."""
        valid_datasets = ["statpearls", "textbooks", "mimic-iv-note", "mimic-iv-bhc"]
        if v not in valid_datasets:
            raise ValueError(
                f"Invalid dataset name: {v}. Must be one of {valid_datasets}."
            )
        return v


class FederationInfo(BaseModel):
    """Federation configuration and metadata (no runtime objects)."""

    benchmark_id: str = Field(..., description="Unique benchmark run identifier")
    data_owners: List[DataOwnerInfo] = Field(
        ..., description="Data owner configurations"
    )
    aggregator: EmailStr = Field(
        ..., description="Aggregator (data scientist) email address"
    )
    network_key: str = Field(..., description="Network key identifier")
    num_data_owners: int = Field(..., ge=1, description="Number of data owners")


class JobProcessingStatus(str, Enum):
    """Enumeration of possible job approval statuses."""

    approved = "approved"
    rejected = "rejected"

    submission_failed = "submission_failed"
    submission_succeeded = "submission_succeeded"
    processing_failed = "processing_failed"


class JobInfo(BaseModel):
    """Information about a submitted job in the federation."""

    job: Optional[Job] = Field(None, description="Syft job object")
    do_email: EmailStr = Field(..., description="Data owner email")
    dataset: str = Field(..., description="Dataset name")
    data_fraction: Optional[float] = Field(None, description="Data fraction used")
    status: JobProcessingStatus = Field(..., description="Job status")
    client: Optional[RDSClient] = Field(None, description="Syft client object")
    error: Optional[str] = Field(None, description="Error message if failed")
    benchmark_id: Optional[str] = Field(None, description="Benchmark run ID")

    class Config:
        arbitrary_types_allowed = True  # Allow Syft objects


class JobProcessingResult(BaseModel):
    """Results from processing submitted jobs."""

    total: int
    num_approved: int
    num_rejected: int
    approved_jobs: List[JobInfo]
    rejected_jobs: List[JobInfo]
    processing_failed_jobs: List[JobInfo]
    approval_rate: float = Field(..., ge=0.0, le=1.0)


class DatasetUploadStatus(str, Enum):
    """Enumeration of dataset upload statuses."""

    success = "success"
    failed = "failed"


class DatasetUploadInfo(BaseModel):
    """Information about a single dataset upload."""

    do_email: EmailStr = Field(..., description="Data owner email")
    dataset_name: str = Field(..., description="Dataset name")
    status: DatasetUploadStatus = Field(..., description="Upload status")
    syft_dataset_name: Optional[str] = Field(None, description="Syft dataset name")
    dataset_object: Optional[Any] = Field(None, description="Syft dataset object")
    corpus_path: Optional[str] = Field(None, description="Path to corpus")
    data_fraction: Optional[float] = Field(None, description="Data fraction")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type if failed")

    class Config:
        arbitrary_types_allowed = True  # Allow Syft objects


class DatasetUploadResult(BaseModel):
    """Results from uploading datasets to data owners."""

    total: int = Field(..., ge=0, description="Total number of uploads")
    success_count: int = Field(..., ge=0, description="Number of successful uploads")
    failure_count: int = Field(..., ge=0, description="Number of failed uploads")
    successful: List[DatasetUploadInfo] = Field(
        default_factory=list, description="Successful uploads"
    )
    failed: List[DatasetUploadInfo] = Field(
        default_factory=list, description="Failed uploads"
    )
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")


class DSServerResult(BaseModel):
    """Results from DS aggregator server execution."""

    status: str = Field(..., description="Execution status (success/failed/error)")
    returncode: Optional[int] = Field(None, description="Process return code")
    stdout: Optional[str] = Field(None, description="Standard output")
    stderr: Optional[str] = Field(None, description="Standard error")
    error: Optional[str] = Field(None, description="Error message if failed")


class FedRAGExecutionResult(BaseModel):
    """Results from FedRAG job execution (DOs and DS)."""

    total_jobs: int = Field(..., ge=0, description="Total number of jobs run")
    successful_jobs: int = Field(..., ge=0, description="Number of successful DO jobs")
    failed_jobs: int = Field(..., ge=0, description="Number of failed DO jobs")
    job_results: List[JobInfo] = Field(..., description="Individual job results")
    ds_server_result: DSServerResult = Field(
        ..., description="DS aggregator server result"
    )
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Job success rate")


class DatasetMetrics(BaseModel):
    """Metrics for a single dataset."""

    total_questions: int = Field(..., ge=0, description="Total questions in dataset")
    answered_questions: int = Field(..., ge=0, description="Questions answered")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy score")
    mean_query_time: Optional[float] = Field(
        None, description="Mean query time in seconds"
    )


class OverallMetrics(BaseModel):
    """Overall aggregated metrics across all datasets."""

    total_questions: int = Field(..., ge=0, description="Total questions")
    total_answered: int = Field(..., ge=0, description="Total answered")
    weighted_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted accuracy"
    )
    mean_query_time: Optional[float] = Field(
        None, description="Mean query time in seconds"
    )


class BenchmarkMetadata(BaseModel):
    """Metadata about the benchmark run."""

    benchmark_id: str = Field(..., description="Unique benchmark identifier")
    start_time: str = Field(..., description="Start time (ISO format)")
    end_time: str = Field(..., description="End time (ISO format)")
    duration_seconds: float = Field(..., ge=0.0, description="Duration in seconds")
    configuration: Dict[str, Any] = Field(..., description="Benchmark configuration")


class FederationMetrics(BaseModel):
    """Federation configuration metrics."""

    num_data_owners: int = Field(..., ge=1, description="Number of data owners")
    data_owners: List[Dict[str, Any]] = Field(..., description="Data owner details")
    aggregator: EmailStr = Field(..., description="Aggregator email")
    network_key: str = Field(..., description="Network key")


class ResultsMetrics(BaseModel):
    """Results metrics containing per-dataset and overall metrics."""

    per_dataset: Dict[str, DatasetMetrics] = Field(
        ..., description="Per-dataset metrics"
    )
    overall: OverallMetrics = Field(..., description="Overall metrics")


class ExecutionMetrics(BaseModel):
    """Execution summary metrics."""

    total_jobs: int = Field(..., ge=0, description="Total jobs")
    successful_jobs: int = Field(..., ge=0, description="Successful jobs")
    failed_jobs: int = Field(..., ge=0, description="Failed jobs")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    ds_server_status: str = Field(..., description="DS server status")


class BenchmarkMetrics(BaseModel):
    """Complete benchmark metrics and results."""

    benchmark_metadata: BenchmarkMetadata = Field(..., description="Benchmark metadata")
    federation: FederationMetrics = Field(..., description="Federation configuration")
    results: ResultsMetrics = Field(..., description="Results metrics")
    execution: ExecutionMetrics = Field(..., description="Execution metrics")
