import pandas as pd

# every_year_path is a dict where keys represent specific years, and values correspond to the file paths for that year
def data_concat(every_year_path):
    def merge_by_path(file_list,var='SEQN'):
        df1 = pd.read_sas(file_list[0],encoding="utf-8")
        for i in range(1,len(file_list)):
            df2 = pd.read_sas(file_list[i],encoding="utf-8")
            df1 = pd.merge(df1, df2, on=var, how='outer')
        return df1
    result=pd.DataFrame()
    for year in every_year_path.keys():
        print("*"*10,"\n",year)
        result=pd.concat([result,merge_by_path(every_year_path[year])], ignore_index=True)
    return result