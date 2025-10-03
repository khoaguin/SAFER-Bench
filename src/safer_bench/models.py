from typing_extensions import Optional, List, Any
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
