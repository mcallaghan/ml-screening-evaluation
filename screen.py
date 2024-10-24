import pandas as pd
import numpy as np
from synergy_dataset import Dataset, iter_datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3
import time
from datetime import timedelta
import typer
import math

class ModelPipeline():
    """
    A sklearn-like model pipeline that can be fit, and that can produce predictions
    """
    def fit(X, y):
        return 

    def predict_proba(X, y):
        return

class ScreenConfig():
    def __init__(self, sample_prop=0.1, batch_prop=0.1):
        """
        Configuration options for the screening simulation

        Parameters
        ----------
        sample_prop : float
            The proportion of the initial dataset that should be included in an initial sample before ML starts
        batch_size : int
            The number of documents that should be 'screened' before re-training a model and making new predictions
        """
        self.sample_prop = sample_prop
        self.batch_prop = batch_prop

class ScreenSimulation():
    """
    An object to manage scr
    """
    
    def __init__(self, df, pipeline, seed, screen_config):
        """
        Parameters
        ----------
        df : pd.Dataset
            A dataframe containing a text variable X, and an inclusion variable y
        pipeline : pipeline
            A sklearn-like model pipeline that can be fit on X and y, and can make predictions for y based on X
        seed : int
            A random seed to ensure results are reproducible
        screen_config : ScreenConfig
            A ScreenConfig object containing options for setting up the screening process
        """
        self.df = df
        self.pipeline = pipeline
        self.seed = seed
        self.screen_config = screen_config
        self.df['screened'] = 0

    def sample(self, seed, conf):
        """
        Simulate screening an initial sample of documents
        """
        sample_ids = self.df.sample(frac=conf.sample_prop, random_state=seed).index
        # Keep adding documents to the sample until we have at least 1 positive example
        while self.df.loc[sample_ids,'y'].sum()==0:
            new_s_ids = self.df[~self.df.index.isin(sample_ids)].sample(1, random_state=seed).index
            sample_ids = sample_ids.union(new_s_ids, sort=False)
        self.df.loc[sample_ids,'screened'] = 1
        self.df.loc[sample_ids,'screened_order'] = np.arange(sample_ids.shape[0])+1
        

    def simulate(self, out_path, run_id, review_id, verbose=False):
        """
        Run a simulation of screening documents with ML prioritisation
        """
        # 'Screen' a sample of documents
        self.sample(self.seed, self.screen_config)
        batch_i = 0
        # As long as there are documents left unscreened
        while self.df.screened.sum() < self.df.shape[0]:
            # Get the indices of screened and not yet screened documents
            train_idx = self.df[self.df['screened']==1].index
            pred_idx = self.df[self.df['screened']==0].index
            
            # Fit a model on the screened documents
            self.pipeline.fit(
                self.df.loc[train_idx, 'X'], 
                self.df.loc[train_idx,'y']
            )
            
            # Make predictions for the unscreened documents
            y_pred = self.pipeline.predict_proba(self.df.loc[pred_idx,'X'])[:,1]
            # Define the batch size (batch_prop) of screened records
            batch_size = math.ceil(len(train_idx)*self.screen_config.batch_prop)
            batch_size = min(batch_size,y_pred.shape[0])
            # Get the *batch_size* documents with the highest predictions
            batch_idx = np.argsort(y_pred)[:-batch_size-1:-1]
            # Get the ids of this batch
            batch_id = pred_idx[batch_idx]
            # Mark this batch as screened and record the order they are screened in
            self.df.loc[batch_id,'screened'] = 1
            self.df.loc[batch_id,'screened_order'] = np.arange(
                train_idx.shape[0],
                train_idx.shape[0] + batch_id.shape[0]
            ) + 1
            # Log the predictions
            self.df.loc[pred_idx,'last_prediction'] = y_pred
            if batch_i==0:
                self.df.loc[pred_idx,'first_prediction'] = y_pred
            # Write predictions to the database
            n_screened = train_idx.shape[0]
            pdf = pd.DataFrame({
                'review_id': review_id,
                'run_id': run_id,
                'n_screened': n_screened,
                'prediction': y_pred,
            })
            pdf.to_parquet('/p/tmp/maxcall/ml-screening/batch_predictions',partition_cols=['review_id','run_id','n_screened'])
            if verbose:
                # Print progress and recall
                print(self.df.loc[self.df['screened']==1].shape[0]-train_idx.shape[0])
                print(self.df.loc[self.df['screened']==1,'y'].sum()/self.df['y'].sum())

            batch_i += 1
        return self.df

def main(version: str):
    """
    Run evaluations testing ML prioritisation across multiple ML pipelines and multiple datasets
    """

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(rank)
        test = False
    except:
        import random
        rank = 0
        print(f'Test mode: randomly assigned {rank}')
        test = True

    db_path = '/p/tmp/maxcall/ml-screening/experiments.sql'
    db_path = 'output_data/experiments.sql'

    con = sqlite3.connect(db_path, timeout=120)
    cur = con.cursor()
    stmt = f'SELECT run_id FROM runs WHERE version="{version}" AND thread="{rank}";'
    cur.execute(stmt)
    res = cur.fetchone()
    if res is None:
        cur.execute('INSERT INTO runs (thread,version) VALUES (?,?)',(rank,version))
        con.commit()
        run_id = cur.lastrowid
        cur.close()
    else:
        run_id = res[0]

    con.close()
    
    pipelines = [
        Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', SVC(probability=True, class_weight='balanced')),
            ]
        )
    ]

    configs = [
        ScreenConfig()
    ]
    
    
    
    for i, d in enumerate(iter_datasets()):
        df = d.to_frame().rename(columns={'label_included':'y'})
        df['X'] = df['title'].astype(str) + ' ' + df['abstract'].astype(str)
        
        if rank==0:
            con = sqlite3.connect(db_path, timeout=120)
            cur = con.cursor()
            stmt = 'INSERT INTO REVIEWS (review_id, review_name, n_records, prevalence) VALUES (?,?,?,?);'
            data = (i,d.name, df.shape[0], df['y'].sum()/df.shape[0])
            cur.execute(stmt, data)
            con.commit()
            review_id = cur.lastrowid
            cur.close()
            con.close()
        review_id = i
        
        for pipe in pipelines:
            for conf in configs:
                ss = ScreenSimulation(df, pipe, rank, conf)
                res = ss.simulate(db_path, run_id, review_id)
                res = res.reset_index()
                res['run_id'] = run_id
                res['review_id'] = review_id
                res = res.rename(columns={
                    'openalex_id': 'rec_id',
                    'y': 'relevant'
                })
                res[[
                    'run_id','rec_id','screened_order','review_id','relevant','first_prediction','last_prediction'
                ]].to_parquet('/p/tmp/maxcall/ml-screening/ordered_records',partition_cols=['review_id','run_id'])
            
            

if __name__ == '__main__':
    start_time = time.monotonic()
    typer.run(main)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
