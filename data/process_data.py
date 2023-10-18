import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data from the files and merge them based on the 'id' column.
    
    Args:
        messages_filepath (str): The file path of the messages CSV file.
        categories_filepath (str): The file path of the categories CSV file.
        
    Returns:
        df (DataFrame): The merged DataFrame containing the data from both input files.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    categories = df.categories.str.split("-|;", expand=True)
    
    category_colnames = categories.iloc[0,::2].values
    
    categories = categories.iloc[:,1::2].astype(int)
    categories.columns = category_colnames
    
    # drop the original categories column from `df`
    df = df.drop(columns="categories")
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True, how='inner')

    return df


def clean_data(df):
    """
    Cleans a given DataFrame
    
    Parameters:
    - df (DataFrame): The input DataFrame to be cleaned.
    
    Returns:
    - df (DataFrame): The cleaned DataFrame.
    """
    # drop duplicates
    df = df.drop_duplicates()
    
    ## Drop classes that have no values
    df = df[np.append(df.columns[:4],df.columns[4:][np.where(df.iloc[:,4:].astype(int).sum() > 1)[0]])]

    ## Drop rows where value other than 0 or 1 occurs
    df = df[df.iloc[:,4:].isin([0,1]).sum(axis=1) == df.iloc[:,4:].shape[1]].reset_index(drop=True) 
    
    return df


def save_data(df, database_filename):
    """
    Save the DataFrame to a SQLite database at the given path.

    Parameters:
        df (DataFrame): The DataFrame to be saved.
        database_filename (str): The filepath of the SQLite database.

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('data', engine, index=False, if_exists='replace')


def main(args):

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(args.messages_filepath, args.categories_filepath))
    
    df = load_data(args.messages_filepath, args.categories_filepath)

    print('Cleaning data...')
    
    df = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(args.database_filepath))
    
    save_data(df.iloc[:args.max_items], args.database_filepath)
    
    print('Cleaned data saved to database!')
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('messages_filepath', type=str,
                        help='The path of the CSV file with the messages')
    parser.add_argument('categories_filepath', type=str,
                        help='The path of the CSV file with the categories')
    parser.add_argument('database_filepath',  type=str,
                        help='The path for the database to be created from the dataset')
    parser.add_argument('--max-items', dest='max_items', action='store', default=None,
                        help='limit number of rows from the datasets to store in database (for testing purposes)')
    
    args = parser.parse_args()
    
    main(args)