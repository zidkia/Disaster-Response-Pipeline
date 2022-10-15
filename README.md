# Disaster Response Pipeline Project
This project is part of Udacity - Data scientists nano degree progrem. 

### The project will includes the foloowing - 

Project Components
There are three components you'll need to complete for this project.

## 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
## 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
## 3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you

Projects Files’:

disaster_messages.csv - data set of the messages  disaster_categories.csv - another data set regarding the categories of the messages

ETL Pipeline Preparation.ipynb - preparing the ETL pipeline before the ML. 
ML Pipeline Preparation.ipynb - preparing the ML pipeline


run.py - main code to run the web application. process_data.py - ETL pipeline for loading, cleaning, and preparing the data for the ML pipeline.
train_classifier.py - ML pipeline used to build our model and present results


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
