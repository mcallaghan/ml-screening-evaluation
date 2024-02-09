import pandas as pd
import numpy as np
from synergy_dataset import Dataset, iter_datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
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
        

    def simulate(self, verbose=False):
        """
        Run a simulation of screening documents with ML prioritisation
        """
        # 'Screen' a sample of documents
        self.sample(self.seed, self.screen_config)
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

            if verbose:
                # Print progress and recall
                print(self.df.loc[self.df['screened']==1].shape[0])
                print(self.df.loc[self.df['screened']==1,'y'].sum()/self.df['y'].sum())
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

    con = sqlite3.connect('output_data/experiments.sql', timeout=20)
    cur = con.cursor()
    cur.execute('INSERT INTO runs (thread,version) VALUES (?,?)',(rank,version))
    con.commit()
    run_id = cur.lastrowid
    cur.close()

    pipelines = [
        Pipeline(
            steps=[
                ('vect', TfidfVectorizer()),
                ('clf', SVC(probability=True, class_weight='balanced')),
            ]
        )
    ]
    
    conf = ScreenConfig()
    
    for d in iter_datasets():
        df = d.to_frame().rename(columns={'label_included':'y'})
        df['X'] = df['title'].astype(str) + ' ' + df['abstract'].astype(str)
        for pipe in pipelines:
            ss = ScreenSimulation(df, pipe, rank, conf)
            res = ss.simulate()
            res = res.reset_index()
            res['run_id'] = run_id
            res['review'] = d.name
            res = res.rename(columns={
                'openalex_id': 'rec_id',
                'y': 'relevant'
            })
            res[[
                'run_id','rec_id','screened_order','review','relevant'
            ]].to_sql(
                'ordered_records', con, if_exists='append', index=False
            )

if __name__ == '__main__':
    start_time = time.monotonic()
    typer.run(main)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
