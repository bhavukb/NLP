import pandas as pd
import numpy as np

def data_augment(df):
    first_three = (df[1] < 4)
    df_to_concat = df[first_three]
    # return pd.concat([df, df_to_concat, df_to_concat, df_to_concat], ignore_index=True)
    return pd.concat([df, df_to_concat], ignore_index=True)
