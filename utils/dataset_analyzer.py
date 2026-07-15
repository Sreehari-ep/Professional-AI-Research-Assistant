from pathlib import Path
import pandas as pd
def load_dataset(path):
    path=Path(path)
    df=pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    df.columns=[str(c).strip() for c in df.columns]; return df
def get_dataset_overview(df):
    return {"rows":len(df),"columns":len(df.columns),"missing_values":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),"numeric_columns":list(df.select_dtypes(include="number").columns),"categorical_columns":list(df.select_dtypes(exclude="number").columns)}
def compare_numeric_by_group(dataframe,value_column,group_column):
    working=dataframe.copy(); working[value_column]=pd.to_numeric(working[value_column],errors="coerce")
    return working.dropna(subset=[value_column,group_column]).groupby(group_column)[value_column].agg(Count="count",Mean="mean",Median="median",Minimum="min",Maximum="max",Standard_Deviation="std").reset_index().round(2)
def calculate_group_difference(comparison_dataframe,group_column,first_group,second_group):
    indexed=comparison_dataframe.set_index(group_column)
    return round(float(indexed.loc[first_group,"Mean"])-float(indexed.loc[second_group,"Mean"]),2)
