# Disaster Response Pipeline Project

A data science project, part of rhe Udacity Data Science Nanodegree program, that focuses on building a machine learning model to classify disaster messages. The project consists of two main components: an ETL (Extract, Transform, Load) pipeline and an ML (Machine Learning) pipeline.

The ETL pipeline is responsible for cleaning and preprocessing the data, which includes merging and splitting the message and category data, removing duplicates, and storing the cleaned data in a SQLite database.

The ML pipeline involves training a multi-output classifier using natural language processing techniques. The pipeline includes feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency) and other custom features, such as counting nouns, verbs, and named entities. The classifier used is XGBoost, which is tuned using GridSearchCV.

The trained model is then saved as a pickle file for future use. The project also includes a web application where users can input disaster messages and get classification results in various categories.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        [Optional] Add `--max_items` flag to limit number of rows from the csv to store in database (for testing purposes).
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to the app directiory 
    `cd app/`

3. Run the following command in the app's directory to run your web app.
    `python run.py` if you have not changed the file names from above
    OR
    `python run.py --database_filepath [database_filepath] --model_filepath [model_filepath]` if you have changed the names
    [Optional] Add `--port` to change the port to serve the app

4. Go to http://0.0.0.0:3001/ or whichever `port` you are running on.


### Project Directory Structure

1. disaster_response_pipeline_project/
    1.1. app/
        1.1.1. templates/
            1.1.1.1. go.html
            1.1.1.2. master.html
        1.1.2. run.py
    1.2. data/
        1.2.1. disaster_categories.csv
        1.2.2. disaster_messages.csv
        1.2.3. process_data.py
        1.2.4. DisasterResponse.db
    1.3. models/
        1.3.1. train_classifier.py
        1.3.2. classifier.pkl
    1.4. README.md