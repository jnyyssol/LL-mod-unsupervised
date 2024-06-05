import polars as pl
import numpy as np
from collections import Counter
#Faster sklearn enabled. See https://intel.github.io/scikit-learn-intelex/latest/
# Causes problems in RandomForrest. We have to use older version due to tensorflow numpy combatibilities
# from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA

from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from loglead.RarityModel import RarityModel
from loglead.OOV_detector import OOV_detector

from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import scipy.sparse
import math
import time

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings


class AnomalyDetection:
    def __init__(self, item_list_col=None, numeric_cols=None, emb_list_col=None, label_col="anomaly", 
                 store_scores=False, print_scores=True):
        self.item_list_col = item_list_col
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.label_col = label_col
        self.emb_list_col = emb_list_col
        self.store_scores = store_scores
        self.print_scores=print_scores
        self.train_vocabulary = None

        
    def test_train_split(self, df, test_frac=0.9, shuffle=True,vec_name="CountVectorizer",unique_test=False):
        # Shuffle the DataFrame
        if shuffle:
            df = df.sample(fraction = 1.0, shuffle=True)
        elif 'start_time' in df.columns:
            df = df.sort('start_time')
        # Split ratio
        test_size = int(test_frac * df.shape[0])

        # Split the DataFrame using head and tail
        self.test_df = df.head(test_size)
        self.train_df = df.tail(-test_size)
        
        if unique_test:
            self.test_df = self.test_df.unique(subset=["e_message_normalized"]) 
        self.prepare_train_test_data(vec_name=vec_name)
        
    def prepare_train_test_data(self, vec_name="CountVectorizer"):
        #Prepare all data for running
        self.X_train, self.labels_train = self._prepare_data(True, self.train_df, vec_name)
        self.X_test, self.labels_test = self._prepare_data(False, self.test_df,vec_name)
        #No anomalies dataset is used for some unsupervised algos. 
        self.X_train_no_anos, _ = self._prepare_data(True, self.train_df.filter(pl.col(self.label_col).not_()), vec_name)
        self.X_test_no_anos, self.labels_test_no_anos = self._prepare_data(False, self.test_df, vec_name)
     
        
    def _prepare_data(self, train, df_seq, vec_name):
        X = None
        labels = df_seq.select(pl.col(self.label_col)).to_series().to_list()

        # Extract events
        if self.item_list_col:
            # Extract the column
            column_data = df_seq.select(pl.col(self.item_list_col))             
            events = column_data.to_series().to_list()
            vectorizer_class = globals()[vec_name]
            # We are training
            if train:
                # Check the datatype  
                if column_data.dtypes[0]  == pl.datatypes.Utf8: #We get strs -> Use SKlearn Tokenizer
                    self.vectorizer = vectorizer_class() 
                elif column_data.dtypes[0]  == pl.datatypes.List(pl.datatypes.Utf8): #We get list of str, e.g. words -> Do not use Skelearn Tokinizer 
                    self.vectorizer = vectorizer_class(analyzer=lambda x: x)
                X = self.vectorizer.fit_transform(events)
                self.train_vocabulary = self.vectorizer.vocabulary_

            # We are predicting
            else:
                X = self.vectorizer.transform(events)

        # Extract lists of embeddings
        if  self.emb_list_col:
            emb_list = df_seq.select(pl.col(self.emb_list_col)).to_series().to_list()
            
            # Convert lists of floats to a matrix
            #emb_matrix = np.array(emb_list)
            emb_matrix = np.vstack(emb_list)
            # Stack with X
            X = hstack([X, emb_matrix]) if X is not None else emb_matrix

        # Extract additional predictors
        if self.numeric_cols:
            additional_features = df_seq.select(self.numeric_cols).to_pandas().values
            X = hstack([X, additional_features]) if X is not None else additional_features

        return X, labels    
        
         
    def train_model(self, model, filter_anos=False):
        X_train_to_use = self.X_train_no_anos if filter_anos else self.X_train
        #Store the current the model and whether it uses ano data or no
        self.model = model
        self.filter_anos = filter_anos
        self.model.fit(X_train_to_use, self.labels_train)

    def predict(self, custom_plot=False):
        #X_test, labels = self._prepare_data(train=False, df_seq=df_seq)
        X_test_to_use = self.X_test_no_anos if self.filter_anos else self.X_test
        if isinstance(self.model, TruncatedSVD):
            predictions = self.TruncatedSVD_reconstruction_predict(X_test_to_use)
        elif isinstance(self.model, SparsePCA):
            predictions = self.SparsePCA_reconstruction_predict(X_test_to_use)
        elif self.model == "randomized_svd":
            predictions = self.randomized_svd_reconstruction_predict(X_test_to_use)
        else:
            predictions = self.model.predict(X_test_to_use)
        #Unsupervised modeles give predictions between -1 and 1. Convert to 0 and 1
        if isinstance(self.model, (IsolationForest, LocalOutlierFactor,KMeans, OneClassSVM)):
            predictions = np.where(predictions < 0, 1, 0)
        df_seq = self.test_df.with_columns(pl.Series(name="pred_normal", values=predictions.tolist()))
        if self.print_scores:
            self._print_evaluation_scores(self.labels_test, predictions, self.model)
        if custom_plot:
            self.model.custom_plot(self.labels_test)
        if self.store_scores:
            self.storage.store_test_results(self.labels_test, predictions, self.model, 
                                            self.item_list_col, self.numeric_cols, self.emb_list_col)
        return df_seq 
       
    def train_LR(self, max_iter=1000):
        self.train_model (LogisticRegression(max_iter=max_iter))
    
    def train_DT(self):
        self.train_model (DecisionTreeClassifier())

    def train_LSVM(self, penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, max_iter=1000):
        self.train_model (LinearSVC(
            penalty=penalty, tol=tol, C=C, dual=dual, class_weight=class_weight, max_iter=max_iter))

    def train_IsolationForest(self, n_estimators=100,  max_samples='auto', contamination="auto",filter_anos=False):
        self.train_model (IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, contamination=contamination), filter_anos=filter_anos)
                          
    def train_LOF(self, n_neighbors=20, max_samples='auto', contamination="auto", filter_anos=True):
        #LOF novelty=True model needs to be trained without anomalies
        #If we set novelty=False then Predict is no longer available for calling.
        #It messes up our general model prediction routine
        self.train_model (LocalOutlierFactor(
            n_neighbors=n_neighbors,  contamination=contamination, novelty=True), filter_anos=filter_anos)
    
    def train_KMeans(self, filter_anos=False):
        self.train_model(KMeans(n_init="auto",n_clusters=2), filter_anos=filter_anos)

    def train_OneClassSVM(self):
        self.train_model(OneClassSVM(max_iter=1000))

    def train_TruncatedSVD(self, filter_anos=False):
        comp = int(len(self.train_vocabulary)*0.95)
        self.train_model(TruncatedSVD(n_components=comp), filter_anos=filter_anos)

    def train_RF(self):
        self.train_model( RandomForestClassifier())

    def train_XGB(self):
        self.train_model(XGBClassifier())
        
    def train_loglizer_PCA(self):
        self.train_model(PCA())

    def train_RarityModel(self, filter_anos=True, threshold=250):
        self.train_model(RarityModel(threshold), filter_anos=filter_anos)
        
    def train_OOVDetector(self, len_col=None, filter_anos=True, threshold=1):
        if len_col == None: 
            len_col = self.item_list_col+"_len"
        self.train_model(OOV_detector(len_col, self.test_df, threshold), filter_anos=filter_anos)
        

    def evaluate_all_ads(self, disabled_methods=[]):
        for method_name in sorted(dir(self)): 
            if (method_name.startswith("train_") 
                and not method_name.startswith("train_model") 
                and method_name not in disabled_methods):
                method = getattr(self, method_name)
                if callable(method):
                    if not self.print_scores:
                        print (f"Running {method_name}")
                    time_start = time.time()
                    method()
                    self.predict()
                    if self.print_scores:
                        print(f'Total time: {time.time()-time_start:.2f} seconds')
        if self.print_scores:
            print("---------------------------------------------------------------")

    def evaluate_with_params(self, models_dict):
        for func_name, params in models_dict.items():
            time_start = time.time()
            func_name = "train_"+func_name
            method = getattr(self, func_name)
            method(**params)
            self.predict()
            print(f'Total time: {time.time()-time_start:.2f} seconds')


    # This function is revamped for the mod
    def _print_evaluation_scores(self, y_test, y_pred, model):
            #Function to find the best F1 score
        def bayesian_optimization(true_labels, anomaly_scores, max_iterations=20):
            t_start = time.time()
            space = [Real(min(anomaly_scores), max(anomaly_scores), name='threshold')]
            
            @use_named_args(space)
            def objective(**params):
                threshold = params['threshold']
                predicted_labels = (anomaly_scores >= threshold).astype(int)
                return -f1_score(true_labels, predicted_labels)

            # Suppress the specific UserWarning about objective re-evaluation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                res_gp = gp_minimize(objective, space, n_calls=max_iterations, random_state=0)
            print(f"F1 optimization time taken {(time.time() - t_start):.4f}")
            return res_gp.x[0], -res_gp.fun
        
        print("Results from model: ", type(model).__name__)
        titlestr = type(self.model).__name__ + " ROC"
        X_test_to_use = self.X_test_no_anos if self.filter_anos else self.X_test
        if isinstance(self.model, (IsolationForest)):
            y_pred = 1 - model.score_samples(X_test_to_use) #lower = anomalous
            print(f"AUCROC: {auc_roc_analysis(y_test, y_pred, titlestr):.4f}")
            print(f"F1: {bayesian_optimization(y_test, y_pred)[1]:.4f}")
        if isinstance(self.model, KMeans):
            y_pred = np.min(model.transform(X_test_to_use), axis=1) #Shortest distance from the cluster to be used as ano score
            print(f"AUCROC: {auc_roc_analysis(y_test, y_pred, titlestr):.4f}")
            print(f"F1: {bayesian_optimization(y_test, y_pred)[1]:.4f}")
        if isinstance(self.model, (RarityModel, OOV_detector, TruncatedSVD)):
            print(f"AUCROC: {auc_roc_analysis(y_test,  model.scores, titlestr):.4f}")
            print(f"F1: {bayesian_optimization(y_test, model.scores)[1]:.4f}")


def auc_roc_analysis(labels, preds, titlestr = "ROC", plot=False):
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, preds)
    # Compute the AUC from the points of the ROC curve
    roc_auc = auc(fpr, tpr)

    if plot:
        # Plot the ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(titlestr)
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc

    