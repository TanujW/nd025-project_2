# Disaster Response Pipeline Project

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
