import pandas as pd
import numpy as np
from copy import deepcopy

def get_table_variables(df):

  df_new = pd.DataFrame(columns = df.column.unique())
  for row in np.sort(df.row.unique()):
    an_array = ['']*df.column.nunique()
    #row = 5
    for col in (df.groupby('column')['center_x'].mean().sort_values().index.values):
      if (len(df.loc[(df.column==col) & (df.row==row)]) > 0):
        an_array[col] = (df.loc[(df.column==col) & (df.row==row)].texts.values[0])
      else:
        an_array[col] = ''

    appended = deepcopy(an_array)

    df_new = df_new.append(pd.Series(appended), ignore_index=True)

  df_new = df_new[[col for col in df.groupby('column')['center_x'].mean().sort_values().index.values]]

  return df_new
 
 
def main():
  path = "/content/drive/My Drive/HACKATHONBBVA_2020/image_Doc2.csv"

  df = pd.read_csv(path)
  df = df.sort_values('row').reset_index(drop=True)

  df_new = get_table_variables(df)
  # df_new.to_csv('/content/drive/My Drive/HACKATHONBBVA_2020/table_variables_doc3.csv')
  df_new.to_csv("table_variables_doc2.csv")
  
