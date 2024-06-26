#
#Separate demo files
import sys
# Suppress ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import loglead.loaders.supercomputers as load_sc
import loglead.loaders.hadoop as load_hadoop
import loglead.loaders.hdfs as load_hdfs
import loglead.loaders.base as load_base

import loglead.enhancer as er, loglead.anomaly_detection as ad
import polars as pl
import time 
import numpy as np

#Adjust full data source
full_data = ""

#List the representations (column names) for anomaly detection
items = ["e_words","e_trigrams","e_event_id"]#,"e_trigrams","e_event_id"]# "e_trigrams","e_alphanumerics"

shuffle_train = True

#List the models as well as possible parameters
models_dict = {
    "IsolationForest": {"filter_anos":True},
    "KMeans": {"filter_anos":True},
    "RarityModel": {"filter_anos":True},
    "OOVDetector": {"filter_anos":True},
}

#### TARKISTA ennen runia filter_anos, CV/tfidf, shuffle, splitit

# Notes:
### Each dataset has their own section where, e.g., you can determine how much data to use. 
### While the code is mostly the same, event and sequence based datasets have some differences 
### Whether the model uses completely unfiltered data (i.e. anomalies in training) can be adjusted with parameter "filter_anos"
### Here normalization is simply turning to lowercase, all numbers to 0s and multiple subsequent 0s to single 0


print("---------- BGL ----------")
frac_data = 0.1
test_frac = 0.95
stime = time.time()
loader = load_sc.BGLLoader(filename=f"{full_data}/bgl/BGL.log")
df = loader.execute()
print("ano", len(df.filter(df["normal"]==False)))
print("normal", len(df.filter(df["normal"]==True)))
df = loader.reduce_dataframes(frac=frac_data)
df = df.filter(pl.col("m_message").is_not_null())
print("time loaded", time.time()-stime)

enhancer =  er.EventLogEnhancer(df)
stime = time.time()
regexs = [('0','\d'),('0','0+')]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time()-stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time()-stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time()-stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time()-stime)
stime = time.time()

df = enhancer.length("e_message_normalized")

sad = ad.AnomalyDetection() 
for item in items:
    print("-----", item, "-----")
    sad.item_list_col = item
    stime = time.time()
    sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vec_name="TfidfVectorizer", unique_test=False)
    print("time split and prepare:", time.time()-stime)
    sad.evaluate_with_params(models_dict)
items = ["e_words"]#,"e_trigrams","e_event_id"]#,"e_trigrams","e_event_id"]# "e_trigrams","e_alphanumerics"


print("---------- HDFS ----------")
frac_data = 0.1
test_frac = 0.95
stime = time.time()
loader = load_hdfs.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                    labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
df = loader.execute()
df = loader.reduce_dataframes(frac=frac_data)
df_seq = loader.df_seq       
print("time loaded", time.time()-stime)

df = df.filter(pl.col("m_message").is_not_null())
enhancer =  er.EventLogEnhancer(df)
df = enhancer.length()


regexs = [('0','\d'),('0','0+')]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time()-stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time()-stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time()-stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time()-stime)
stime = time.time()

seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seq)
print("ano", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==False)))
print("normal", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==True)))
seq_enhancer.seq_len()
seq_enhancer.start_time()


sad = ad.AnomalyDetection()
for item in items:
    print("-----", item, "-----")
    if item != "e_event_id":
        seq_enhancer.tokens(item)
    else:
        seq_enhancer.events(item)
    sad.item_list_col = item

    stime = time.time()
    sad.test_train_split(seq_enhancer.df_seq, test_frac=test_frac, shuffle=shuffle_train, vec_name="TfidfVectorizer")
    print("time split and prepare:", time.time()-stime)
    sad.evaluate_with_params(models_dict)


print("---------- Thunderbird ----------")
frac_data = 0.003
test_frac = 0.95
stime = time.time()
loader = load_sc.ThuSpiLibLoader(filename=f"{full_data}/thunderbird/tb1.log", split_component=False)
df = loader.execute()
print("ano", len(df.filter(df["normal"]==False)))
print("normal", len(df.filter(df["normal"]==True)))
df = loader.reduce_dataframes(frac=frac_data)
df = df.filter(pl.col("m_message").is_not_null())
enhancer =  er.EventLogEnhancer(df)
print("time loaded", time.time()-stime)


enhancer =  er.EventLogEnhancer(df)
stime = time.time()
regexs = [('0','\d'),('0','0+')]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time()-stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time()-stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time()-stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time()-stime)
stime = time.time()

df = enhancer.length("e_message_normalized")

sad = ad.AnomalyDetection() 
for item in items:
    print("-----", item, "-----")
    sad.item_list_col = item
    
    stime = time.time()
    sad.test_train_split (df, test_frac=test_frac, vec_name="TfidfVectorizer")
    print("time split and prepare:", time.time()-stime)
    
    sad.evaluate_with_params(models_dict)
    

print("---------- spirit ----------")
frac_data = 0.003
test_frac = 0.95
stime = time.time()
loader = load_sc.ThuSpiLibLoader(filename=f"{full_data}/spirit/spirit2.log", split_component=False)
df = loader.execute()
print("ano", len(df.filter(df["normal"]==False)))
print("normal", len(df.filter(df["normal"]==True)))
df = loader.reduce_dataframes(frac=frac_data)
df = df.filter(pl.col("m_message").is_not_null())
enhancer =  er.EventLogEnhancer(df)
print("time loaded", time.time()-stime)


enhancer =  er.EventLogEnhancer(df)
stime = time.time()
regexs = [('0','\d'),('0','0+')]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time()-stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time()-stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time()-stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time()-stime)
stime = time.time()

df = enhancer.length("e_message_normalized")

sad = ad.AnomalyDetection() 
for item in items:
    print("-----", item, "-----")
    sad.item_list_col = item
    
    stime = time.time()
    sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vec_name="TfidfVectorizer")
    print("time split and prepare:", time.time()-stime)
    
    sad.evaluate_with_params(models_dict)
    
    
print("---------- liberty ----------")
frac_data = 0.003
test_frac = 0.95
stime = time.time()
loader = load_sc.ThuSpiLibLoader(filename=f"{full_data}/liberty/liberty2.log", split_component=False)
df = loader.execute()
print("ano", len(df.filter(df["normal"]==False)))
print("normal", len(df.filter(df["normal"]==True)))
df = loader.reduce_dataframes(frac=frac_data)
df = df.filter(pl.col("m_message").is_not_null())
enhancer =  er.EventLogEnhancer(df)
print("time loaded", time.time()-stime)


enhancer =  er.EventLogEnhancer(df)
stime = time.time()
regexs = [('0','\d'),('0','0+')]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time()-stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time()-stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time()-stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time()-stime)
stime = time.time()

df = enhancer.length("e_message_normalized")

sad = ad.AnomalyDetection() 
for item in items:
    print("-----", item, "-----")
    sad.item_list_col = item
    
    stime = time.time()
    sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vec_name="TfidfVectorizer")
    print("time split and prepare:", time.time()-stime)
    
    sad.evaluate_with_params(models_dict)


print("---------- Hadoop ----------")
frac_data = 1
test_frac = 0.5
stime = time.time()
loader = load_hadoop.HadoopLoader(filename=f"{full_data}/hadoop/",
                                            filename_pattern  ="*.log",
                                            labels_file_name=f"{full_data}/hadoop/abnormal_label.txt")
df = loader.execute()
df = loader.reduce_dataframes(frac=frac_data)

df_seq = loader.df_seq       
print("time loaded", time.time()-stime)
df = df.filter(pl.col("m_message").is_not_null())

enhancer =  er.EventLogEnhancer(df)
df = enhancer.length()

regexs = [('0','\d'),('0','0+')]
df = enhancer.normalize(regexs, to_lower=True)
print("time normalized", time.time()-stime)
stime = time.time()
df = enhancer.trigrams("e_message_normalized")
print("time trigrams", time.time()-stime)
stime = time.time()
df = enhancer.words("e_message_normalized")
print("time words", time.time()-stime)
stime = time.time()
df = enhancer.parse_drain()
print("time parse", time.time()-stime)
stime = time.time()

seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seq)
print("ano", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==False)))
print("normal", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==True)))
seq_enhancer.seq_len() #OOVD uses data from the df for faster calculations
seq_enhancer.start_time()

sad = ad.AnomalyDetection()
for item in items:
    print("-----", item, "-----")
    if item != "e_event_id":
        seq_enhancer.tokens(item)
    else:
        seq_enhancer.events(item)
    sad.item_list_col = item

    stime = time.time()
    sad.test_train_split (seq_enhancer.df_seq, test_frac=test_frac, shuffle=shuffle_train, vec_name="TfidfVectorizer")
    print("time split and prepare:", time.time()-stime)

    sad.evaluate_with_params(models_dict)
