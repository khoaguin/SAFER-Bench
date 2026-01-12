"""Specialty classification for medical text chunks.

This module provides keyword-based classification of clinical text
into medical specialties for specialty-based data distribution.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Core 6 specialties with comprehensive keyword lists
SPECIALTY_KEYWORDS: Dict[str, List[str]] = {
    "cardiology": [
        "cardiac",
        "heart",
        "coronary",
        "ekg",
        "ecg",
        "echocardiogram",
        "arrhythmia",
        "atrial fibrillation",
        "afib",
        "myocardial",
        "infarction",
        "stemi",
        "nstemi",
        "angina",
        "pacemaker",
        "defibrillator",
        "icd",
        "cardiovascular",
        "hypertension",
        "htn",
        "chf",
        "heart failure",
        "cardiomyopathy",
        "valvular",
        "aortic",
        "mitral",
        "tricuspid",
        "pericardial",
        "endocarditis",
        "catheterization",
        "angiogram",
        "stent",
        "cabg",
        "bypass",
        "troponin",
        "bnp",
        "ejection fraction",
    ],
    "oncology": [
        "cancer",
        "tumor",
        "tumour",
        "malignant",
        "malignancy",
        "chemotherapy",
        "chemo",
        "radiation therapy",
        "radiotherapy",
        "metastasis",
        "metastatic",
        "carcinoma",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "myeloma",
        "neoplasm",
        "neoplastic",
        "oncologist",
        "biopsy",
        "staging",
        "remission",
        "palliative",
        "immunotherapy",
        "targeted therapy",
        "adjuvant",
        "neoadjuvant",
        "pet scan",
        "ca-125",
        "psa",
        "cea",
        "afp",
    ],
    "neurology": [
        "stroke",
        "cva",
        "tia",
        "seizure",
        "epilepsy",
        "neurological",
        "brain",
        "cerebral",
        "cns",
        "central nervous system",
        "peripheral neuropathy",
        "neuropathy",
        "parkinson",
        "alzheimer",
        "dementia",
        "multiple sclerosis",
        "ms",
        "meningitis",
        "encephalitis",
        "headache",
        "migraine",
        "eeg",
        "electroencephalogram",
        "lumbar puncture",
        "csf",
        "mri brain",
        "ct head",
        "weakness",
        "paralysis",
        "hemiparesis",
        "aphasia",
        "ataxia",
        "tremor",
        "gcs",
        "glasgow coma",
        "consciousness",
        "cranial nerve",
    ],
    "pulmonology": [
        "lung",
        "pulmonary",
        "respiratory",
        "pneumonia",
        "copd",
        "emphysema",
        "bronchitis",
        "asthma",
        "ventilator",
        "intubation",
        "extubation",
        "oxygen",
        "hypoxia",
        "hypoxemia",
        "dyspnea",
        "shortness of breath",
        "sob",
        "chest x-ray",
        "cxr",
        "ct chest",
        "bronchoscopy",
        "pleural",
        "effusion",
        "thoracentesis",
        "chest tube",
        "ards",
        "pe",
        "pulmonary embolism",
        "pneumothorax",
        "atelectasis",
        "spirometry",
        "pft",
        "fev1",
        "fvc",
        "sputum",
        "nebulizer",
        "inhaler",
        "bipap",
        "cpap",
    ],
    "gastroenterology": [
        "liver",
        "hepatic",
        "hepatitis",
        "cirrhosis",
        "gi",
        "gastrointestinal",
        "bowel",
        "intestinal",
        "colon",
        "colonoscopy",
        "endoscopy",
        "egd",
        "stomach",
        "gastric",
        "esophagus",
        "esophageal",
        "gerd",
        "reflux",
        "ulcer",
        "gi bleed",
        "hematemesis",
        "melena",
        "hematochezia",
        "pancreatitis",
        "pancreatic",
        "cholecystitis",
        "gallbladder",
        "biliary",
        "ascites",
        "varices",
        "ibd",
        "crohn",
        "colitis",
        "diverticulitis",
        "obstruction",
        "ileus",
        "alt",
        "ast",
        "bilirubin",
        "albumin",
        "inr",
        "meld",
    ],
    "nephrology": [
        "kidney",
        "renal",
        "dialysis",
        "hemodialysis",
        "peritoneal dialysis",
        "creatinine",
        "gfr",
        "egfr",
        "bun",
        "ckd",
        "chronic kidney disease",
        "aki",
        "acute kidney injury",
        "arf",
        "esrd",
        "end stage renal",
        "transplant",
        "nephropathy",
        "glomerulonephritis",
        "proteinuria",
        "hematuria",
        "oliguria",
        "anuria",
        "uremia",
        "electrolyte",
        "potassium",
        "hyperkalemia",
        "hypokalemia",
        "sodium",
        "hyponatremia",
        "hypernatremia",
        "acidosis",
        "alkalosis",
        "fistula",
        "av graft",
        "catheter",
    ],
    "general": [
        "sepsis",
        "septic",
        "bacteremia",
        "cellulitis",
        "abscess",
        "infection",
        "infectious",
        "antibiotic",
        "antibiotics",
        "mrsa",
        "vre",
        "osteomyelitis",
        "endocarditis",
        "meningitis",
        "pneumonia",  # Often infectious
        "uti",
        "urinary tract infection",
        "pyelonephritis",
        "fever",
        "febrile",
        "leukocytosis",
        "bacteriuria",
        "wound infection",
        "surgical site infection",
    ],
}

# Threshold for infection override: if general/infectious score > this fraction of top score, classify as general
# 0.75 = general must be at least 75% of the top specialty score to override
INFECTION_OVERRIDE_THRESHOLD = 0.75


# Section weights for specialty classification
# Higher weights for sections that indicate primary diagnosis
SECTION_WEIGHTS: Dict[str, float] = {
    "Discharge Diagnosis": 10.0,
    "Chief Complaint": 8.0,
    "Brief Hospital Course": 3.0,
    "History Of Present Illness": 2.0,
    "Assessment And Plan": 5.0,
    # Default weight for other sections
    "_default": 1.0,
}


def _count_keywords(text: str, keywords: List[str]) -> int:
    """Count keyword occurrences in text."""
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        if len(keyword) <= 3:
            pattern = rf"\b{re.escape(keyword)}\b"
            count += len(re.findall(pattern, text_lower))
        else:
            count += text_lower.count(keyword.lower())
    return count


def classify_chunk(
    content: str, return_scores: bool = False
) -> str | Tuple[str, Dict[str, float]]:
    """Classify a text chunk into a medical specialty.

    Uses keyword matching with case-insensitive search.
    Returns the specialty with the highest keyword match count.

    Args:
        content: The text content to classify
        return_scores: If True, also return the score dict

    Returns:
        The specialty name with highest match count, or "general" if no matches.
        If return_scores=True, returns (specialty, scores_dict)
    """
    content_lower = content.lower()
    scores: Dict[str, float] = {}

    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        count = 0
        for keyword in keywords:
            # Use word boundary matching for short keywords to avoid false positives
            if len(keyword) <= 3:
                pattern = rf"\b{re.escape(keyword)}\b"
                count += len(re.findall(pattern, content_lower))
            else:
                count += content_lower.count(keyword.lower())
        scores[specialty] = float(count)

    # Get specialty with highest score
    if max(scores.values()) == 0:
        result = "general"
    else:
        result = max(scores, key=scores.get)

    if return_scores:
        return result, scores
    return result


def classify_note(chunks: List[Dict]) -> str | Tuple[str, Dict[str, float]]:
    """Classify a full clinical note using section-weighted scoring.

    Weights sections by their diagnostic importance:
    - Discharge Diagnosis: 10x weight
    - Chief Complaint: 8x weight
    - Brief Hospital Course: 3x weight
    - History Of Present Illness: 2x weight
    - Other sections: 1x weight

    Args:
        chunks: List of chunk dicts, each with 'content' and 'metadata.section'

    Returns:
        The specialty name with highest weighted score
    """
    scores: Dict[str, float] = {spec: 0.0 for spec in SPECIALTY_KEYWORDS}

    for chunk in chunks:
        content = chunk.get("content", "")
        section = chunk.get("metadata", {}).get("section", "")
        weight = SECTION_WEIGHTS.get(section, SECTION_WEIGHTS["_default"])

        # Count keywords for each specialty in this chunk
        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            count = _count_keywords(content, keywords)
            scores[specialty] += count * weight

    # Get specialty with highest score
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def classify_note_file(
    note_file: Path, return_scores: bool = False
) -> str | Tuple[str, Dict[str, float]]:
    """Classify a clinical note file using section-weighted scoring.

    Args:
        note_file: Path to .jsonl file containing note chunks
        return_scores: If True, also return the score dict

    Returns:
        The specialty name with highest weighted score
    """
    chunks = []
    with open(note_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    scores: Dict[str, float] = {spec: 0.0 for spec in SPECIALTY_KEYWORDS}

    for chunk in chunks:
        content = chunk.get("content", "")
        section = chunk.get("metadata", {}).get("section", "")
        weight = SECTION_WEIGHTS.get(section, SECTION_WEIGHTS["_default"])

        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            count = _count_keywords(content, keywords)
            scores[specialty] += count * weight

    if max(scores.values()) == 0:
        result = "general"
    else:
        # Get top specialty (excluding general for initial comparison)
        non_general_scores = {k: v for k, v in scores.items() if k != "general"}
        top_specialty = max(non_general_scores, key=non_general_scores.get)
        top_score = non_general_scores[top_specialty]

        # Apply infection override rule:
        # If general/infectious score > threshold * top_score, classify as general
        general_score = scores.get("general", 0)
        if top_score > 0 and general_score > INFECTION_OVERRIDE_THRESHOLD * top_score:
            result = "general"
        else:
            result = top_specialty

    if return_scores:
        return result, scores
    return result


def generate_specialty_mapping(
    chunks_dir: Path,
    output_path: Optional[Path] = None,
    use_section_weights: bool = True,
) -> Dict[str, str]:
    """Process all note files in a directory and generate specialty mapping.

    Args:
        chunks_dir: Path to directory containing .jsonl chunk files
        output_path: Optional path to save the mapping JSON
        use_section_weights: If True, use section-weighted scoring (recommended)

    Returns:
        Dictionary mapping note_id (filename stem) to specialty
    """
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunk_files = sorted(chunks_dir.glob("*.jsonl"))
    if not chunk_files:
        raise ValueError(f"No .jsonl files found in {chunks_dir}")

    logger.info(f"Processing {len(chunk_files)} note files from {chunks_dir}")
    logger.info(f"Using section-weighted scoring: {use_section_weights}")

    mapping: Dict[str, str] = {}
    specialty_counts: Counter = Counter()

    for chunk_file in chunk_files:
        # Extract note_id from filename (without .jsonl extension)
        note_id = chunk_file.stem

        try:
            if use_section_weights:
                # Use section-weighted classification
                specialty = classify_note_file(chunk_file)
            else:
                # Fall back to simple keyword counting
                with open(chunk_file, "r", encoding="utf-8") as f:
                    content = f.read()
                specialty = classify_chunk(content)

            mapping[note_id] = specialty
            specialty_counts[specialty] += 1

        except Exception as e:
            logger.warning(f"Error processing {chunk_file}: {e}")
            mapping[note_id] = "general"
            specialty_counts["general"] += 1

    # Log distribution
    logger.info("Specialty distribution:")
    for specialty, count in specialty_counts.most_common():
        percentage = (count / len(mapping)) * 100
        logger.info(f"  {specialty}: {count} ({percentage:.1f}%)")

    # Save mapping if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        logger.success(f"Specialty mapping saved to {output_path}")

    return mapping


def get_chunks_by_specialty(
    mapping: Dict[str, str],
    specialty: str,
) -> List[str]:
    """Get all chunk IDs belonging to a specific specialty.

    Args:
        mapping: Dictionary mapping chunk_id to specialty
        specialty: Target specialty name

    Returns:
        List of chunk_ids for the specified specialty
    """
    return [chunk_id for chunk_id, spec in mapping.items() if spec == specialty]


def load_specialty_mapping(mapping_path: Path) -> Dict[str, str]:
    """Load a specialty mapping from JSON file.

    Args:
        mapping_path: Path to the specialty_mapping.json file

    Returns:
        Dictionary mapping chunk_id to specialty
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Specialty mapping not found: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)
