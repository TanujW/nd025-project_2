import argparse
import re

import json
import plotly
import pandas as pd
import numpy as np
import nltk

# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import dill
from sqlalchemy import create_engine


app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    
    global df, model
    
    # load data
    engine = create_engine(f'sqlite:///{args.database_filepath}')
    df = pd.read_sql_table('data', engine.connect())

    # load model
    with open(args.model_filepath, "rb") as f:
        model = dill.load(f)
    
    
    app.run(host='0.0.0.0', port=args.port, debug=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run Flask server for webapp")
    parser.add_argument('--database-filepath', dest='database_filepath', action='store', default="../data/DisasterResponse.db", required=False,
                        help='The path of the disaster messages database (DEFAULT="../data/DisasterResponse.db")')
    parser.add_argument('--model-filepath', dest='model_filepath', action='store', default="../models/classifier.pkl", required=False,
                        help='The path of the saved classifier model (DEFAULT="../models/classifier.pkl")')
    parser.add_argument('--port', dest='port', action='store', default=3001, required=False,
                        help='[Optional] Port to run the app on (DEFAULT=3001)')
    
    args = parser.parse_args()
    
    
    main()