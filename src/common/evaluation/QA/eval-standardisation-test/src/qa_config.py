import sys
from pathlib import Path

USE_HEIGHT = True
USE_WEIGHT = False

assert USE_HEIGHT != USE_WEIGHT, "Please enable either HEIGHT or WEIGHT"

sys.path.append(str(Path(__file__).parents[0]))

if USE_HEIGHT:
    from qa_config_height import *  # noqa: F401, F403, E402
elif USE_WEIGHT:
    from qa_config_weight import *  # noqa: F401, F403, E402
else:
    raise Exception("Please choose either weight or height!")
