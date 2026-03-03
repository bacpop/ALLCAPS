# Deprecated scripts

The following scripts have been moved here because they are superseded by
newer implementations.  They are kept for reference only — do **not** import
from this directory in production code.

| Script | Replaced by |
|--------|-------------|
| `train_transformer.py` | `trihead/train_trihead_transformer.py` (3-head model with genogroup) |
| `infer_transformer.py` | `trihead/infer_trihead_transformer.py` (3-head model support via ModelRegistry) |
| `process_query.py` | `trihead/process_trihead_query.py` (trihead support, eval/scan modes, OpenMax) |
| `novel_detection.py` | `trihead/process_trihead_query.py` + `openmax.py` (consolidated) |
