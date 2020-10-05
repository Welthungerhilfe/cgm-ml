import os
from pathlib import Path

import pandas as pd

REPO_DIR = Path(os.getcwd()).parents[5]

## error margin on various ranges
EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2, 2.5, 3, 4, 5, 6]

CODE_TO_SCANTYPE = {
    '100': '_front',
    '101': '_360',
    '102': '_back',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}

def calculate_performance(code, df_mae):
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in EVALUATION_ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error']<=acc) & (df_mae_filtered['error']>=-acc)]
        if len(df_mae_filtered):
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        # print(f"Accuracy {acc:.1f} for {code}: {accuracy}")
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = EVALUATION_ACCURACIES
    return df_out
