import os

ROOT = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(ROOT, "models")

INFO_PREFIX = ">  [ info ] "
ERROR_PREFIX = "> [warning] "
WARNING_PREFIX = "> [ error ] "

INPUT_KEY = "input"
LATENT_KEY = "latent"
PREDICTIONS_KEY = "predictions"
LABEL_KEY = "labels"
