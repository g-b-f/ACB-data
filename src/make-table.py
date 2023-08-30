import pipeline as ppl
import pandas as pd
ppl.args['verbose']=0


def getPath(file): # Returns the absolute path to a file, given a path relative to script
    import os
    path = os.path.dirname(os.path.abspath(__file__)) # Gets current path of file
    filepath = os.path.join(path,file) # Sets path relative to current file
    return filepath

def mergeDF(df1, df2,diff=False):
    df_out=df1.merge(df2,on='Name',how='outer') # how='outer'
    print(f"{(len(df1)+len(df2))- len(df_out)} out of {len(df2)} rows merged")
    if diff:
        mask= df1["Name"].isin(df_out["Name"])
        display(df1[~mask])
    return df_out

lozDict={
"ACB":"Ancelin2006",
"ARS":"Rudolph2008",
"ADS":"Carnahan2006",
"ABC":"Ancelin2006",
"AAS":"Ehrt2010",
"ALS":"Sittironnarit2011",
"Generic Name":"Name"
}

df=pd.read_csv(getPath("data/lozano.csv"))
df=df.rename(columns=lozDict)
# special case for an odd name from Bishara:
df["Name"]=df["Name"].replace({"acetaminophen/dichloralphenazone/isometheptene": "acetaminophen"})

bish=pd.read_csv(getPath("data/bishara.csv"))
bish=bish.rename(columns=lambda x: x[-1])


bish2={"Name":[],"Bishara2020":[]}

for col in bish.columns:
    vals=bish[col].dropna().str.lower()
    bish2["Name"].extend(vals)
    bish2["Bishara2020"].extend([int(col) for _ in range(len(vals))])

bish2=pd.DataFrame(bish2)
df=mergeDF(df,bish2)

ACBcalc=pd.read_csv(getPath("data/ACBcalc.csv"))
ACBcalc["Name"]=ACBcalc["Name"].str.lower()
ACBcalc=ACBcalc.rename(columns={"Score": "ACBcalc"})
df=mergeDF(df,ACBcalc)

df=df.drop_duplicates()
uDF=ppl.unify_score(df,name="Name").rename(columns={"Score":"MajVote"})
df=mergeDF(df,uDF)

df=df.set_index("Name")
df=df.astype("Int8") # Int8 (with capital 'I') is nullable int type
df.to_csv(getPath("../ACB_table.csv"))