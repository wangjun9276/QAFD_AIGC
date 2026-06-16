# Validation status

Completed checks:

- All Python files pass syntax compilation.
- `test_IQAG.py --help` loads successfully when declared dependencies are present.
- Synthetic binary-folder discovery was tested.
- Image loading, preprocessing, CLIP token shapes, and DataLoader collation were tested.
- Binary ACC/AUC/AP calculation was tested on a known example.

An end-to-end numerical inference run was not possible because the uploaded ZIP
did not include the trained IQAG checkpoint, the OpenAI CLIP checkpoint, or the
benchmark image folders. The script now checks these paths explicitly and
reports clear errors when they are missing.
