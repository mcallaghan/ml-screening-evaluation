import pyarrow as pa
import pyarrow.parquet as pq
from buscarpy import retrospective_h0
import time
import typer
import sqlite3
import pandas as pd
from itertools import product
import numpy as np

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

    db_path = 'output_data/experiments.sql'
    con = sqlite3.connect(db_path)
    rdf = pd.read_sql_query('SELECT * FROM reviews',con)

    cur = con.cursor()
    stmt = f'SELECT run_id FROM runs WHERE version="{version}" AND thread="{rank}";'
    cur.execute(stmt)
    res = cur.fetchone()
    run_id = res[0]

    batch_size = 10

    for i, row in rdf.iterrows():

        print(row['review_id'])
    
        table = pq.read_table(
            '/p/tmp/maxcall/ml-screening/ordered_records', 
            filters=[
                ('review_id', '=', row['review_id']),
                ('run_id', '=', run_id)
            ]
        ).to_pandas().sort_values('screened_order').reset_index(drop=True)
        for recall_target in [0.8,0.9,0.95,0.99]:
            pdf = pd.DataFrame(
                retrospective_h0(
                    table['relevant'], table.shape[0], 
                    batch_size=batch_size, plot=False, recall_target=recall_target,
                    confidence_level=0.99
                )
            ).rename(columns={'p': f'p_{recall_target}'})
            if f'p_{recall_target}' in table.columns:
                table = table.drop(columns=f'p_{recall_target}')
            table = table.merge(
                pdf, left_on='screened_order', right_on='batch_sizes', how='left'
            ).drop(columns='batch_sizes').ffill()
        table.to_parquet(
            '/p/tmp/maxcall/ml-screening/ordered_records',
            partition_cols=['review_id','run_id'],
            existing_data_behavior='delete_matching'
        )

    # models = pq.read_table(
    #     '/p/tmp/maxcall/ml-screening/llm_preds'
    # )['model'].unique().dictionary.to_pylist()

    # combinations = list(product(rdf.review_name, models))
    # combinations = [x for i,x in enumerate(combinations) if i%100==rank]
    # for review_name, model in combinations:
    #     llm_df = pq.read_table(
    #         f'/p/tmp/maxcall/ml-screening/llm_preds',
    #         filters=[
    #             ('review', '=', review_name),
    #             ('model', '=', model)
    #         ],
    #         columns=['y','py','pn','review','model','openalex_id']
    #     ).to_pandas()
    #     print(review_name)
    #     print(model)
    #     print(llm_df.shape)
    #     llm_df['pred'] = llm_df['py'] - llm_df['pn']
    #     llm_df = llm_df.sort_values('pred', ascending=False).reset_index()
    #     llm_df['screened_order'] = np.arange(llm_df.shape[0])
    #     for recall_target in [0.8,0.9,0.95,0.99]:
    #         pdf = pd.DataFrame(
    #             retrospective_h0(
    #                 llm_df['y'], llm_df.shape[0], 
    #                 batch_size=batch_size, plot=False, recall_target=recall_target,
    #                 confidence_level=0.99
    #             )
    #         ).rename(columns={'p': f'p_{recall_target}'})
    #         llm_df = llm_df.merge(
    #             pdf, left_on='screened_order', right_on='batch_sizes', how='left',
    #         ).drop(columns='batch_sizes').ffill()
    #     llm_df = llm_df.set_index('openalex_id')
    #     llm_df.to_parquet(
    #         '/p/tmp/maxcall/ml-screening/llm_preds',
    #         partition_cols=['review','model'],
    #         existing_data_behavior='delete_matching'
    #     )

if __name__ == '__main__':
    start_time = time.monotonic()
    typer.run(main)
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))