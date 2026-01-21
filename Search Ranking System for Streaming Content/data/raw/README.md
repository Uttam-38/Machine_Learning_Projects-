# Raw Data Directory

This directory stores **raw, immutable datasets** used by the project.

## Contents
- Downloaded datasets (e.g., MovieLens archives)
- Unprocessed source files exactly as obtained

## Usage Guidelines
- Files in this directory **should not be modified**
- This directory is **excluded from version control**
- Data is generated locally via:
  ```bash
  python -m src.download_data
