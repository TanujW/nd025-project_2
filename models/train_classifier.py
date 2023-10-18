import argparse
import dill, pickle

## Data wrangling libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text as sql_text

## NLP libraries
import re
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('universal_tagset')
nltk.download('words')
nltk.download('maxent_ne_chunker')

## ML libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBClassifier

def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Parameters:
        database_filepath (str): The file path of the SQLite database.

    Returns:
        X (Series): The input data for the model.
        Y (DataFrame): The target data for the model.
        category_names (list): The category names of the target data.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql(sql=sql_text("SELECT * FROM data"), con=engine.connect())
    X = df.message
    Y = df.iloc[:,4:]
    return X, Y, Y.columns


## Helper functions

# Custom transformer for extracting number of nouns and verbs
class NounAndVerbCounter(BaseEstimator, TransformerMixin):
    import nltk
    def get_pos_tags(self, text):
        return nltk.pos_tag(self.tokenizer(text), tagset='universal')
    
    def count_verbs(self, text):
        return np.count_nonzero(np.ravel(self.get_pos_tags(text)) == "VERB")
    
    def count_nouns(self, text):
        return np.count_nonzero(np.ravel(self.get_pos_tags(text)) == "NOUN")
    
    def __init__(self, tokenizer=nltk.wordpunct_tokenize):
        self.tokenizer = tokenizer
    
    def fit(self, X, y=None):
        return self
        
    def transform(self,X):
        df = pd.DataFrame()
        df['nouns'] = pd.Series(X).apply(self.count_nouns)
        df['verbs'] = pd.Series(X).apply(self.count_verbs)
        return df
   
   
# Custom transformer for extracting number of named entities
class NamedEntityCounter(BaseEstimator, TransformerMixin):
    import nltk
    def count_named_entities(self, text):
        ner = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)), binary=True)
        return len([n for n in ner if not isinstance(n, tuple)])
    
    def fit(self, X, y=None):
        return self
        
    def transform(self,X):
        df = pd.DataFrame()
        df['ne'] = pd.Series(X).apply(self.count_named_entities)
        return df

def tokenize(text):
    """
    Tokenizes the input text after removing URLs and lemmatizing the words.

    Parameters:
    - text (str): The text to be tokenized.

    Returns:
    - clean_tokens (list): A list of lemmatized tokens from the text.
    """
    import nltk
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "__link__")

    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds and returns a machine learning model pipeline for text classification.
    
    Returns:
        model (GridSearchCV): A grid search cross-validation pipeline that combines feature extraction, transformation, and classification.
    """
    
    pipeline = Pipeline([
    ('feature',FeatureUnion([
        ('embedding_pipeline',Pipeline([
             ('vec_count', CountVectorizer(tokenizer=tokenize)),
            ('tf-idf', TfidfTransformer())
        ])),
        ('noun_verb_counter', NounAndVerbCounter(tokenizer=tokenize)),
        ('ner_counter', NamedEntityCounter())
    ])),
    ('clf', MultiOutputClassifier(XGBClassifier(random_state=42, n_jobs=-1, eval_metric='aucpr'), n_jobs=-1))
])
    
    params = {'clf__estimator__max_depth': [2, 4, 6, 8],
              'clf__estimator_learning_rate': [0.01, 0.1, 0.5],
              'clf__estimator__gamma': [0.1, 0.5, 1],
              'clf__estimator__n_estimators': [50, 100, 200]}
    
    model = GridSearchCV(pipeline, params, verbose=2)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate and display the performance of the classifier.
    
    Args:
        model (Sklearn model/pipeline): The trained machine learning model.
        X_test (array-like): The test data.
        Y_test (array-like): The true labels for the test data.
        category_names (list): The names of the categories.
        
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    print("\033[1mOverall Performance\033[0m")
    print('-'*20)
    print(classification_report(Y_test.values.ravel(), y_pred.ravel()))

    print("\033[1mPer Category\033[0m")
    print("-"*20)
    
    for (label,true),(_,pred) in zip(Y_test.items(),pd.DataFrame(y_pred, columns=Y_test.columns).items()):
        print("\033[1mCategory:"+label+"\033[0m")
        print(classification_report(true, pred))


def save_model(model, model_filepath):
    """
    Save a machine learning pipeline including custom functions to a file using dill.

    Parameters:
        model (Sklearn model/pipeline): The trained machine learning model to save.
        model_filepath (str): The file path to save the model to.

    Returns:
        None
    """
    ## Using dill to save custom functions used in pipeline
    with open(model_filepath, "wb") as f:
        dill.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    
    print('Loading data...\n    DATABASE: {}'.format(args.database_filepath))
    X, Y, category_names = load_data(args.database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(args.model_filepath))
    save_model(model, args.model_filepath)

    print('Trained model saved!')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('database_filepath', type=str, default="../data/DisasterResponse.db",
                        help='The path of the disaster messages database (e.g., ../data/DisasterResponse.db)')
    parser.add_argument('model_filepath', type=str, default="classifier.pkl",
                        help='The path to save the model (e.g., classifier.pkl)')
    
    args = parser.parse_args()
    
    
    main(args)