from typing_extensions import Optional, List
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
