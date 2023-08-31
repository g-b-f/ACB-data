import pipeline as ppl
import pandas as pd
ppl.args['verbose']=0

def getPath(file): # Returns the absolute path to a file, given a path relative to script
    import os
    path = os.path.dirname(os.path.abspath(__file__)) # Gets current path of file
    filepath = os.path.join(path,file) # Sets path relative to current file
    return filepath

def mergeDF(df1, df2,diff=False):
    df_out=df1.merge(df2,on='Name',how='outer')
    print(f"{(len(df1)+len(df2))- len(df_out)} out of {len(df2)} rows merged")
    if diff: # optionally display unmerged rows
        mask= df1["Name"].isin(df_out["Name"])
        display(df1[~mask])
    return df_out

df=pd.read_csv(getPath("data/lozano.csv"))
# special case for an odd name from Bishara:
df["Name"]=df["Name"].replace({"acetaminophen/dichloralphenazone/isometheptene": "acetaminophen"})

bish=pd.read_csv(getPath("data/bishara.csv"))
bish=bish.rename(columns=lambda x: x[-1]) # 'Drugs with AEC score of x' -> 'x'

bish2={"Name":[],"Bishara2020":[]}

for score in bish.columns:
    drugName=bish[score].dropna().str.lower()
    bish2["Name"].extend(drugName)
    bish2["Bishara2020"].extend([int(score) for _ in drugName])

bish2=pd.DataFrame(bish2)
df=mergeDF(df,bish2)

ACBcalc=pd.read_csv(getPath("data/ACBcalc.csv"))
ACBcalc["Name"]=ACBcalc["Name"].str.lower()
ACBcalc=ACBcalc.rename(columns={"Score": "ACBcalc"})
df=mergeDF(df,ACBcalc)

df=df.drop_duplicates()
uDF=ppl.unify_score(df,name="Name").rename(columns={"Score":"MajVote"})
df=mergeDF(df,uDF)
uDF=ppl.unify_score(df,name="Name",votingAlg=ppl.kieselTest).rename(columns={"Score":"kiesel"})
df=mergeDF(df,uDF)

df=df.set_index("Name")
df=df.astype("Int8") # Int8 (with capital 'I') is nullable int type
df.to_csv(getPath("../ACB_table.csv"))