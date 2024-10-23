import sqlite3
import typer
from synergy_dataset import Dataset, iter_datasets
import json

def main(delete: int=0):
    # Set up database
    con = sqlite3.connect('/p/tmp/maxcall/ml-screening/experiments.sql')
    con = sqlite3.connect('output_data/experiments.sql')
    create_runs = '''
    CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY, 
    exp_started datetime default current_timestamp, 
    version TEXT,
    thread INT);
    '''
    create_or = '''
    CREATE TABLE IF NOT EXISTS ordered_records (
        id INTEGER PRIMARY KEY,
        rec_id TEXT,
        screened_order INTEGER,
        first_prediction FLOAT,
        last_prediction FLOAT,
        review_id INTEGER,
        relevant INTEGER,
        run_id INTEGER
    );
    '''
    create_batch_preds = '''
    CREATE TABLE IF NOT EXISTS batch_predictions (
        id INTEGER PRIMARY KEY,
        review_id INTEGER,
        prediction FLOAT,
        n_screened INTEGER,
        run_id INTEGER
    );
    '''
    create_reviews = '''
    CREATE TABLE IF NOT EXISTS reviews (
        review_id INTEGER PRIMARY KEY,
        review_name TEXT,
        n_records INTEGER,
        prevalence FLOAT
    )
    '''
    cur = con.cursor()
    if delete>0:
        print('dropping tables')
        cur.execute('DROP TABLE IF EXISTS runs')
        cur.execute('DROP TABLE IF EXISTS ordered_records')
        cur.execute('DROP TABLE IF EXISTS batch_predictions')
        cur.execute('DROP TABLE IF EXISTS reviews')
        cur.execute("VACUUM")
        con.commit()
    cur.execute(create_runs)
    cur.execute(create_or)
    #cur.execute(create_batch_preds)
    cur.execute(create_reviews)
    res = cur.execute('PRAGMA journal_mode=WAL')
    
    with open('pipelines.json','r') as f:
        pipelines = json.load(f)
        print(pipelines)
    

    cur.close()
    return


if __name__ == '__main__':
    typer.run(main)
