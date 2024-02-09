import sqlite3
import typer

def main(delete: int=0):
    # Set up database
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
        review TEXT,
        relevant INTEGER,
        run_id
    );
    '''
    cur = con.cursor()
    if delete>0:
        print('dropping tables')
        cur.execute('DROP TABLE IF EXISTS runs')
        cur.execute('DROP TABLE IF EXISTS ordered_records')
    cur.execute(create_runs)
    cur.execute(create_or)
    return


if __name__ == '__main__':
    typer.run(main)
