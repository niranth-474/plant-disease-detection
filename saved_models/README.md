# Local model artifacts

Place the trained `best_model.pth` checkpoint and its ordered `class_names.json` file in this directory. Run `python -m model.train` from the project root to generate both files. The checkpoint contains model weights, the class count, and class labels.

Do not commit trained model weights or the PlantVillage dataset. Keep the checkpoint and class-name file from the same training run; the prediction service verifies that they match.
