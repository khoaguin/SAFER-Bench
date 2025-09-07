## Datasets

### StatPearls
Downloaded and processed based on https://github.com/adap/flower/tree/main/examples/fedrag/data
- **Type**: Medical reference articles
- **Strength**: Comprehensive, peer-reviewed, educational
- **Language**: Formal, structured, educational
- **Size**: 9,300 articles
- **Use for**: Factual medical knowledge
- **Status**: ✅ Available

### Textbooks
Downloaded and processed based on https://github.com/adap/flower/tree/main/examples/fedrag/data
- **Type**: Educational material
- **Strength**: Foundational concepts, systematic coverage
- **Language**: Formal, detailed explanations
- **Size**: 18 books
- **Use for**: Core medical principles
- **Status**: ✅ Available

### MIMIC-IV-Ext-BHC
Pre-segmented version available from https://physionet.org/content/labelled-notes-hospital-course/1.2.0/
- **Type**: Pre-segmented clinical notes
- **Strength**: Pre-chunked, cleaned, structured sections
- **Language**: Clinical narratives with clear headers
- **Size**: 270,033 notes
- **Use for**: Pre-processed for ML/RAG
- **Status**: ⏳ Planned for integration

### MIMIC-IV-Note
Raw notes downloaded from https://physionet.org/content/mimic-iv-note/2.2/
- **Type**: Real clinical narratives
- **Strength**: Actual medical practice, diverse cases, temporal data
- **Language**: Clinical shorthand, abbreviations, informal
- **Size**: 2.3M+ notes
- **Use for**: Testing real-world applicability
- **Status**: ✅ Available