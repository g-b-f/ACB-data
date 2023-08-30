import os
# HPC = os.environ.get('HPC',False) # Set flag if we're in an HPC environment
# HPC=True

# mm_pg_gold_therapy.csv:
# 2850189 pats
# 693644222 rows
# 243.367 rows per pat average

def main():
    """Main pipeline. Will execute when the program is run at top level, i.e, not imported"""
    import pandas as pd
    chk("starting",timer=False)

    bnfCSV=inputCSV("data/new_bnf_lkp.csv")
    bnfDict=BNF_to_dict(bnfCSV)
    lit_dfs=inputCSV("data/lozano.csv")
    acb=unify_score(lit_dfs,bnfDict)
    chk("making acbDict",3)
    acb=acb[acb['Score']!=0] # drop drugs with 0 acb score
    acb=dict(zip(acb['Code'],acb['Score'])) # create dict of Code: Score
    # save(acb,"acbDict")
    chk()

    fileIn = inputCSV("data/heads/bighead_pres.csv",usecols=["patid","fullbnf","age_days"], 
    dtype={"patid":'int64',"fullbnf":str,"age_days":'uint16'})#,nrows=3*10**4)#,nrows=7*10**6) # should be roughly 10% of all patients


    # fileIn = inputCSV("data/mm_pg_gold_therapy.csv",usecols=["patid","fullbnf","age_days"], 
    # dtype={"patid":'int64',"fullbnf":str,"age_days":'uint16'},nrows=3*10**4)#,nrows=7*10**6) # should be roughly 10% of all patients

    windowOut=run_window(fileIn)

    save(windowOut,"test2_windowOut")

    attrib=inputCSV("data/patient/patients.csv",usecols=["patid",'gender','ethnicity','eth_group','imd_decile'],
    dtype={"patid":'uint64',"gender":"category","ethnicity":"category","eth_group":"category","imd_decile":"category"})

    window_df,pat=cluster_prep(windowOut,acb)
    save((window_df,pat),"test2_window_df_pat")

    lab_df=clustering(window_df, pat, 10)
    save(lab_df,"test2_lab_df")

    tab_dict=describeClust(lab_df, attrib)
    save(tab_dict,"test2_tab_dict")

    df_dict=x2_analyse(tab_dict)
    save(df_dict,"test2_df_dict")

    chk("finished",timer=False)




####################################################

args={"debug":False, "verbose":0, "NSLOTS":1, "window":183, "skip":1, "singlecore":False, "highmem":False,"presFile":"/data/WHRI-Bioinformatics/NIHR_AIM/Projects/Gabriel/data/heads/bighead_pres.csv"}

def get_args():
    """
    Runs just before :func:`main()`, getting command line arguments.
    
    Notes
    ----------
    See :ref:`args` for more information on the specific arguments that can be passed
    """

    import argparse
    import os
    parser = argparse.ArgumentParser(prog='ACB Pipeline',description='A pipeline that takes patient data and creates longitudinal clusters')
    parser.add_argument('-d','--debug',help='Make temp files persist',action='store_true',default=False)
    parser.add_argument('-v','--verbose',help='Increase verbosity. Can be passed up to 3 times', action='count', default=0)
    parser.add_argument('-n','--NSLOTS',help="Number of cores",default=os.environ.get('NSLOTS',1),type=int)
    parser.add_argument('-w','--window',help="Length of window",default=183,type=int) # 183 days = 6 months
    parser.add_argument('-s','--skip',help="Amount to skip after each window",default=1,type=int)
    # parser.add_argument('-c','--clusters',help="Number of clusters",default=10,type=int)
    parser.add_argument('-z','--singlecore',help="Sliding window algorithm in single core mode",action='store_true',default=False)
    parser.add_argument('-m','--highmem',help="Run window algorithm faster but use more memory",action='store_true',default=False)
    parser.add_argument('-f','--presfile',help="path to prescriptions file")

    args.update(vars(parser.parse_args()))
    
### From lib.py ###

startTime = 0
def chk(task=None,min_verbose=1,*,timer=True):
    """
    Displays status of python code.

    Parameters
    ----------
    task: str, optional
        String describing the current task.
        Passing None will end the current timer without starting a new one
    min_verbosity: int, optional
        minimum verbosity level required to display
    timer: bool, optional
        Whether to time the current task
    Notes
    -----
    Do :func:`chk()` (i.e, with no input) at end of program,
    or before statements that will print.

    Examples
    ----------

    >>> import time
    >>> chk("sleeping for a long time")
    >>> time.sleep(5)
    >>> chk("sleeping for a short time")
    >>> time.sleep(0.3)
    >>> chk()
    12:00:00 sleeping for a long time... done, 5.0 seconds.
    12:00:05 sleeping for a short time... done, 300 milliseconds.
    """

    if min_verbose>args['verbose'] and task is not None:
        return

    import time
    global startTime

    if startTime!=0 and timer:
        diff=time.time()-startTime
        if diff>60*60:
            print(f"done, {diff/60**2:.1f} hours.",flush=True)
        elif diff>60:
            print(f"done, {diff/60:.1f} minutes.",flush=True)
        elif diff<0.5:
            print(f"done, {diff*1000:.0f} milliseconds.",flush=True)
        else:
            print(f"done, {diff:.1f} seconds.",flush=True)

    if task is None: # End current timer without starting a new one
        startTime=0
    else:
        clockTime=time.strftime("%H:%M:%S", time.localtime())
        if timer:
            print(f"{clockTime} {task}... ",end='',flush=True)
            startTime=time.time()
        else:
            print(f"{clockTime} {task} ",flush=True)


def inputCSV(path,delim=',',**kwargs):
    """
    Reads csv or similar using Pandas.
    Convenience function that allows for relative paths.

    Parameters
    ----------
    path: str
        Relative file path of file
    delim: str, optional
        delimiter of file. Default is ``,``

    Returns
    ----------
    df: |df|
        parsed DataFrame

    Notes
    ----------
    kwargs are passed to :func:`pandas.read_csv() <pandas:pandas.read_csv>`
    """
    from pandas import read_csv
    from os.path import abspath, getsize

    name=path.split('/')[-1]
    name=name.split('.')[0]
    filepath=abspath(path)
    min_verbose=2 if getsize(filepath) > 10**9 else 3 # increase priority of message if file greater than 1GB
    min_verbose=1 if getsize(filepath) > 10**10 else min_verbose # increase priority of message if file greater than 10GB
    chk(f"reading {name}",min_verbose=min_verbose)
    try:
        df=read_csv(filepath,sep=delim,**kwargs)
    except FileNotFoundError: # can happen if passed an absolute path
        df=read_csv(path,sep=delim,**kwargs)
    chk()
    return df

def save(obj,name):
    """
    Pickles an object

    Parameters
    ----------
    obj: object
        Object to be pickled
    name: str
        name to give pickle file
    """
    import pickle
    chk(f"pickling {name}",3)
    path="/data/home/bt22048/pickles"# if HPC else "g:/My Drive/QMUL/MSc-Final-Project/Programs/output"
    with open(f"{path}/{name}.pickle",'wb') as fo:
        pickle.dump(obj,fo)
    chk()

def load(name):
    """
    Unpickles a file

    Parameters
    ----------
    name: str
        name of pickle file
    
    Returns
    ----------
    obj: object
        Unpickled object
    """
    import pickle
    chk(f"unpickling {name}",3)
    path="/data/home/bt22048/scratch/pickles"
    with open(f"{path}/{name}.pickle",'rb') as fi:
        obj=pickle.load(fi)
    chk()
    return obj


### from collate_acb.ipynb ###

def BNF_to_dict(bnfCSV):
    """
    Converts a bnf csv to a mapping dictionary

    Parameters
    ----------
    bnfCSV: DataFrame
        |df| of BNF codes and names

    Returns
    -------
    mapDict: dict
        Dictionary mapping drug names to BNF codes
    """
    chk("creating dictionary from BNF")
    import pandas as pd
    bnfName=bnfCSV["BNF_Chemical_Substance"].str.lower() # make lowercase
    trimCodes=bnfCSV["BNF_Presentation_Code"].str[:9] # trim to first 9 numbers, to get chemical substance
    bnfName=bnfName.str.split(r"[\s/]",expand=True,regex=True)[0] # first part of bnfName (for cases such as citalopram hydrochloride)
    mapDict=dict(zip(bnfName,trimCodes)) # create from names to codes to mapping dictionary
    chk()
    return mapDict


def kieselTest(arr):
    """
    From an array of different scores for the same drug,
    reach consensus through the algorithm outlined by Kiesel et al. (2018) [1]_

    Parameters
    ---------- 
    arr: array_like
        array of potential scores for the same drug

    Returns
    -------
    score: int or None
        Agreed upon score if possible, None otherwise

    References
    ----------
    .. [1] Kiesel, E.K., Hopf, Y.M. and Drey, M. (2018)
      "An anticholinergic burden score for German prescribers: Score development",
      BMC Geriatrics, 18(1).
      doi: `10.1186/s12877-018-0929-6 <doi.org/10.1186/s12877-018-0929-6>`_.
    """

    import pandas as pd
    
    # If drug scored by two or more lists and score differs by 1 point, drug gets higher score
    counts=pd.Series(arr).value_counts()
    if counts.iloc[0]<2: # If most common score has less than 2 votes,
        return None # it is invalid
    firstScore=counts.index[0] # get most common score
    try:
        second=0
        second=counts.index[1]
        third=counts.index[2]
        return None # if third exists, score is invalid and can't be used
    except IndexError: # If second or third index aren't found, that's a good thing
        if abs(firstScore-second)<=1: # If difference between most common and second most common<=1,
            return int(firstScore)
    
def majVote(arr):
    """
    From an array of different scores for the same drug,
    reach consensus through simple majority voting

    Parameters
    ---------- 
    arr: array_like
        array of potential scores for the same drug

    Returns
    -------
    score: int or None
        Agreed upon score if possible, |None| otherwise
    """

    import pandas as pd

    try:
        scores= pd.Series(arr).value_counts()
        return int(scores.index[0])
    except:
        return None


def unify_score(lit_dfs,bnfDict=None,votingAlg=majVote,name="Generic Name"):
    """
    Return a pandas DataFrame of scores, voted upon using an algorithm.

    Parameters
    ---------- 
    lit_dfs: DataFrame or iterable containing DataFrames
        one or more pandas |dfs|
    bnfDict: dict or None
        dictionary mapping drug names to codes. If None, function will skip mapping
    votingAlg: function,optional
        function to use to vote on scores. Must receive an iterable of potential scores,
        and return None if unable to reach consensus
    name: str, optional
        Name of column containing drug names


    Returns
    -------
    score: |df|
        DataFrame of agreed upon scores

    """
    chk("creating unified score",timer=args['verbose']==1)
    import pandas as pd
    from numpy import isnan

    
    # if only a single df passed instead of an iterable, make it a singleton tuple
    lit_dfs=(lit_dfs,) if type(lit_dfs)== pd.DataFrame else lit_dfs
    
    df_out=[]
    for n,df in enumerate(lit_dfs):
        chk(f"scoring df {n}",3)
        df=df.set_index(name)
        df=df[df.notnull().any(axis=1)] # select rows where at least one column > 0

        allDrugs=[] # all drugs, for testing
        voted=[] # drugs which reached consensus
        for name, row in df.iterrows():
            row = list(filter(lambda x:not isnan(x),row)) # remove NaNs
            allDrugs.append(row)
            val=votingAlg(row)
            if val is not None:
                voted.append((name,val))
        chk()
        testName="kiesel" if votingAlg is kieselTest else "majority voting" if votingAlg is majVote else "unknown algorithm"
        chk(f"using {testName}",3,timer=False)
        chk(f"{len(voted)}/{len(allDrugs)} reached consensus",2,timer=False)

        
        ##### Creating CSV #####

        out=pd.DataFrame(voted,columns=["Name","Score"]) # Create dataframe of calculated scores
        if bnfDict is not None:
            out["Code"]=out["Name"].map(bnfDict) # lookup from names to codes
        outLen=len(out)
        out=out.dropna()

        chk(f"{outLen-len(out)} rows with unknown codes dropped",2,timer=False)
        df_out.append(out)
        chk()
    return(pd.concat(df_out))

### from sliding_window.py ###


def batch_func(gp): # calls the c library
    """
    Used by :func:`run_window()` when calling the C windowing program

    Parameters
    ----------
    gp: |df|
        DataFrame containing values for a single patient
    """
    import c_lib
    patid = int(gp['patid'].iloc[0]) # patid is constant within groups, so only first entry required
    drugID = gp['drugID_numeric'].to_numpy(dtype=int)
    age_days = gp['age_days'].to_numpy(dtype=int)
    c_lib.ltc_iter(patid,drugID,age_days)
    return True


def run_window(fileIn):
    """
    Prepares and run auxiliary files for running the windowing C file
    

    Parameters
    ----------
    fileIn: str
        input file
    Returns
    ----------
    windowOut: |df|
        DataFrame created by the C function

    Notes
    ----------
    If debug mode on, resultant file will be saved in pipeline_xxxx/slideWindow.csv

    Only unix-like systems are supported for this function

    """

    import pandas as pd
    import subprocess
    import numpy as np
    import sysconfig
    import os
    import tempfile
    from sys import platform
    plat=platform.lower()
    assert plat.startswith('linux') or plat.startswith('darwin'), "Only unix-like systems are supported for this function"
    pd.options.mode.chained_assignment = None # disable warning


    chk("windowing",timer=args['verbose']==1)

    chk("processing input file",2)
    fileIn["drugID"]=fileIn["fullbnf"].str[:9] # first nine chars of BNF are individual drug ID
    chk("dropping duplicates",3)
    fileIn=fileIn.drop_duplicates(["patid", "age_days", "drugID"]) # drop duplicate lines

    chk("mapping drugID to drugID_numeric",2)
    uniques=fileIn['drugID'].unique()
    drug_dict_fwd = {drug:number for number, drug in enumerate(uniques)}
    fileIn['drugID_numeric'] = fileIn['drugID'].map(drug_dict_fwd) # map from alphanumeric ID to numeric ID
    chk()

    with tempfile.TemporaryDirectory() as tempDir: # will delete tempDir after use unless...
        if args['debug']: # ... debug option is active
            tempDir=tempfile.mkdtemp(prefix="pipeline_",dir=os.getcwd())
        chk(f"created directory {tempDir}", min_verbose=1 if args['debug'] else 3, timer=False)
        chk("creating LUT",3)
        with open(tempDir+"/iterLUT.h",'wt') as fo: # make C LUT header
            arr=''.join(f'"{x}",' for x in uniques)
            arr="char *LUT[]={"+arr[:-1]+"};"
            fo.write(arr)

        chk("building c library",2)
        py_path=sysconfig.get_paths()['include']
        np_path=np.get_include()

        gcc=['gcc', 'py_c.c', '-O3', '-march=native', '-Werror', '-Wall', '-std=gnu11', '-fPIC', '-shared', '-o','c_lib.so',
        '-I',py_path,'-I', np_path,'-I', tempDir,
        '-D' ,f'FILENAME="{tempDir}/slideWindow.csv"',"-D" ,f"WINDOW_SIZE={args['window']}", "-D", f"SKIP_AMOUNT={args['skip']}"]

        if not args['singlecore']:
            gcc.append('-D')
            gcc.append('MULTICORE')
            # use same (random) suffix as tempDir, so semaphores don't interfere:
            gcc.append('-D')
            gcc.append(f'LTC_SEM="/ltc_iter_mutex_{tempDir.split("/")[-1]}"')

        subprocess.run(gcc)
        chk()
        import c_lib

        for d in [fileIn,drug_dict_fwd,uniques,gcc,arr]:
            del(d) # remove unused stuff

        filepath=f"{tempDir}/slideWindow.csv"

        chk("making csv header",3)
        with open(filepath,'wt') as fo: # add csv header
            fo.write("patid,age_days,drugID\n")

        chk("splitting data",2)
        groups=[data for pat,data in fileIn.groupby('patid')] # maybe refactor this later to just pass the groupby object, and split later
        l_group=len(groups)
        chk(f"Running {l_group} groups on {1 if args['singlecore'] else args['NSLOTS']} cores",2)

        if args['singlecore']: # simple for loop is less prone to error
            for gp in groups:
                batch_func(gp)
        else:
            from multiprocessing import Pool
            c_lib.ltc_iter_semUnlink() # remove semaphore, in case it wasn't removed properly last time
            with Pool(args['NSLOTS']) as p:
                if args['highmem']:
                    p.map(batch_func, groups)
                else:
                    p.imap(batch_func, groups)
                p.close()
                p.join() # wait for all jobs to finish
                c_lib.ltc_iter_semUnlink() # remove semaphore
        
        windowOut=inputCSV(filepath,dtype={"patid":np.int64,"age_days":np.int32,"drugID":str})
        chk()
        
        return windowOut

### from clust_iter.py ###

def cluster_prep(window_df,acbDict,kernel=None,patLen=2):
    """
    Prepares the data for clustering

    Parameters
    ---------- 
    window_df: DataFrame
        |df| containing patients, ID of prescription, and age at prescription
    acbDict: dict
        Dictionary mapping drugs to scores
    kernel: iterable, optional
        Kernel used for smoothing via linear convolution.
        Set to |None| to not smooth
    patLen: int, optional
        Minimum number of patients per group

    
    Returns
    ----------
    window_df: list
        List of scores for each patient, convolved with :obj:`kernel`
    pat: |arr|
        List of patients, in same order as :obj:`window_df`

    """

    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    chk("Preparing data for clustering",timer=args['verbose']==1)
    
    chk("mapping",2)
    window_df['Score']=window_df['drugID'].map(acbDict) # map from BNFcodes to scores
    window_df=window_df.dropna()[['patid','age_days','Score']]
    window_df['Score']=window_df['Score'].astype(int)

    chk("scoring",2)
    window_df=window_df.groupby(['patid','age_days']).sum() # get sum of scores in each window
    window_df = window_df.reset_index([0,1])
    window_df=window_df.groupby('patid').filter(lambda x: len(x) >=patLen)
    pat,window_df=zip(*window_df.groupby("patid"))
    pat=np.array(pat)

    chk("convolving",2)
    window_df=[np.convolve(a['Score'].to_numpy(),kernel) for a in window_df] # get 'Score' array for each pat
    pat=np.array(pat)
    chk()

    return window_df, pat

def clustering(window_df, pat, n_clusters,*,limit=None,metric="dtw",**kwargs):
    """
    Creates time series clusters using :class:`TimeSeriesKMeans <tslearn:tslearn.clustering.TimeSeriesKMeans>`

    Parameters
    ----------
    window_df: iterable of |arr|
        Scores for each patient
    pat: array-like
        Patient IDs
    n_clusters: int
        number of clusters
    metric: {"euclidean", "dtw", "softdtw"}, optional
        Metric to be used for both cluster assignment and barycenter computation.
        If "dtw", DBA is used for barycenter computation
    limit: int, optional
        Use at most this many patients in calculation

    Returns
    ----------
    lab_df: |df|
        DataFrame of patients, each labelled with their predicted cluster

    Notes
    ----------
    ``metric= "euclidean"`` does not work for datasets without equal-length time series,
    such as those used in this pipeline

    kwargs are passed to :class:`TimeSeriesKMeans <tslearn:tslearn.clustering.TimeSeriesKMeans>`

    
    """
    from scipy.stats import chisquare
    from tslearn.utils import to_time_series_dataset
    from tslearn.clustering import TimeSeriesKMeans
    # from tslearn.clustering import KernelKMeans
    # from tslearn.clustering import silhouette_score
    from itertools import product
    import pandas as pd
    import numpy as np
    from sklearn.utils import shuffle
    
    # gives pretty dubious estimations, but better than nothing.
    def time_func(samples,n_clusters,limit):
        d={'Samples': [69716.01171538091, 0.7902720838797705, -1.6361495340296348],
        'Clusters': [22.798245359616875, 52.61347498831894],
        'Limit': [0.7223911973573456, 8.252312398600417]} # calculated from previous iterations
        
        lim=len(window_df) if limit is None else limit
        samp_time=d['Samples'][0]/(d['Samples'][1]*samples)+d['Samples'][2]
        clust_time=(d['Clusters'][0]*n_clusters)+d['Clusters'][1]
        lim_time=(d['Limit'][0]*lim)+d['Limit'][1]
        return(samp_time+clust_time+lim_time)*(16/args['NSLOTS'])
        
    pred_time=time_func(args['window'],n_clusters,limit)
    eta=f"{pred_time/60**2:.2g} hours" if pred_time>60**2 else f"{pred_time/60:.2g} minutes"
    chk(f"limit: {limit}, clusters: {n_clusters}, ETA {eta}",1,timer=args['verbose']<3)

    if limit is not None:
        pat,window_df=shuffle(pat, window_df, n_samples=limit)

    window_df = to_time_series_dataset(window_df)

    model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,n_jobs=args['NSLOTS'],verbose=args['verbose']>3,**kwargs)
    labels = model.fit_predict(window_df)
    save(model,f"new_tsModel_{n_clusters}clust_{len(pat)}pat")
    chk("creating lab_df",3)
    lab_df=pd.DataFrame({"patid":pat,"Label":labels})
    lab_df['Count']=lab_df['Label'].map(lab_df['Label'].value_counts()) # get counts of each label
    chk()
    return lab_df

### from describe_clusters.ipynb ###

def describeClust(clust, attrib, threshold=0):
    """
    Describes the clusters by finding the contents of each

    Parameters
    ----------
    lab_df: DataFrame
        |df| of patients, each labelled with their predicted cluster
    attrib: DataFrame
        |df| of patients and their corresponding attributes
        such as Gender, Ethnicity, etc
    threshold: int, optional
        Ignore clusters smaller than this threshold

    Returns
    ----------
    tab_dict: dict
        Dictionary of |dfs|, in the form {attribute: df}

    """
    import pandas as pd
    chk("describing clusters",timer=args['verbose']==1)
    clust=clust[clust["Count"]>threshold] # mask off entries below threshold

    clust=clust.set_index('patid')
    attrib=attrib.set_index('patid')
    clust=clust.join(attrib).dropna()
    del(attrib)
    clust=clust.drop('Count',axis=1).set_index('Label')

    l=[]
    lab_count=clust.index.value_counts()
    tab_dict={}
    for col in clust.columns: # for each attribute
        chk(f"calculating for {col}",3)
        df={'Label':[],'Label_size':[]}
        all_attrib_names=list(clust[col].value_counts().index) # all possible groups within a given attribute
        df.update({k:[] for k in all_attrib_names})

        for count,lab in zip(lab_count, lab_count.index): # for each label containing that attribute
            df['Label'].append(lab)
            df['Label_size'].append(count)
            attributes=clust.loc[lab][col].value_counts() # count of members

            # Not all labels will have every attribute, thus we have to pad with empty values for non-found attributes: 
            extras=pd.Series([0 for _ in all_attrib_names],index=all_attrib_names) # fill with empty values
            extras=extras.drop(attributes.index) # drop all those that are found, leaving only non found with empty values
            attrib=pd.concat([attributes,extras])
            
            for attrib_name,attrib_count in zip(attrib.index, attrib):
                df[attrib_name].append(attrib_count)

        df=pd.DataFrame(df)
        tab_dict.update({col:df})
    chk()

    return tab_dict

def x2_analyse(tab_dict,sigLevel=0.05,relative=False):
    """
    Describes the clusters by finding the contents of each

    Parameters
    ----------
    tab_dict: dict
        Dictionary of |dfs|, in the form {attribute: df}
    sigLevel: float, optional
        Threshold at which P can be considered significant

    Returns
    ----------
    df_dict: dict
        Dictionary of |dfs|, in the form {attribute: df}

    """
    import pandas as pd
    import numpy as np
    from scipy.stats import chisquare

    chk("Chi square analysis",timer=args['verbose']==1)

    sigCount=0
    df_dict={}
    for attrib,df in tab_dict.items():
        chk(f"analysing {attrib}",3)
        totals=[df[i].sum() for i in df.columns[2:]]
        freqs=[i/df['Label_size'].sum() for i in totals]

        pList=[]
        x2List=[]
        exp=[]

        for idx,row in df.iterrows():
            expected=np.array([i*row["Label_size"] for i in freqs])
            # X2 only valid if expected freq > 5, population > 15, sum of frequencies = 1
            if np.all(expected>5) and row["Label_size"] > 15:
                try:
                    x2,p=chisquare(row[2:],expected)
                except ValueError: # raised if sum of observed and expected don't agree
                    x2,p=(np.nan,np.nan)
            else:
                x2,p=(np.nan,np.nan)
            pList.append(p)
            x2List.append(x2)
            if relative:
                exp.append(expected)

        df['X2']=x2List
        df['P']=pList
        df['significant']=df['P']<sigLevel

        if relative:
            exp=np.array(exp).T # flip cols/ rows
            for n,col in enumerate(df.columns[2:-3]):
                df[col]=(df[col]-exp[n])/exp[n]

        df_dict.update({attrib:df})


        chk()
        sigCount= sigCount +(1 if df['significant'].any() else 0)
        msg=f"{attrib} is significant for at least one label" if df['significant'].any() else f"no sig difference for {attrib}"
        chk(msg,2,timer=False)

    chk()
    chk(f"{sigCount} significant attributes found at P<{sigLevel}",timer=False)

    return df_dict

#############################
if __name__ == "__main__": # ensure this is right at the bottom
    get_args()
    main()