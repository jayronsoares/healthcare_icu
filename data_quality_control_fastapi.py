"""
This example defines a FastAPI app and adds a single route at /data_quality_checks/ that expects a CSV file in the request body. 
The app uses the run_data_quality_checks() function to run all the data quality checks on the uploaded data and returns a list of results as a JSON response. 
You can send a POST request to the /data_quality_checks/ endpoint with a CSV file in the request body to perform the data quality checks.
"""
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def check_accuracy(df):
    # Check for accuracy by ensuring that all data is within acceptable ranges
    # Here we assume that all columns are numeric
    for col in df.columns:
        if df[col].max() > 100 or df[col].min() < 0:
            return False
    return True

def check_completeness(df):
    # Check for completeness by ensuring that there are no missing values in the dataset
    return df.isnull().sum().sum() == 0

def check_consistency(df):
    # Check for consistency by ensuring that all columns have the same data type
    return len(set(df.dtypes)) == 1

def check_timeliness(df):
    # Check for timeliness by ensuring that the data is not older than 30 days
    today = datetime.now()
    data_date = df['date_column'].max() # assuming a date column exists
    return (today - data_date).days < 30

def check_relevance(df):
    # Check for relevance by ensuring that the dataset contains the expected columns
    expected_columns = ['col1', 'col2', 'col3']
    return set(df.columns) == set(expected_columns)

def check_validity(df):
    # Check for validity by ensuring that all values are within acceptable ranges
    # Here we assume that column 'col1' should have values between 0 and 10
    return df['col1'].between(0, 10).all()

def check_precision(df):
    # Check for precision by ensuring that all columns have the expected number of decimal places
    expected_decimal_places = {'col1': 2, 'col2': 3}
    for col in expected_decimal_places:
        if df[col].apply(lambda x: len(str(x).split('.')[1])).max() != expected_decimal_places[col]:
            return False
    return True

def check_uniqueness(df):
    # Check for uniqueness by ensuring that there are no duplicate rows in the dataset
    return len(df) == len(df.drop_duplicates())

def run_data_quality_checks(df):
    # Run all data quality checks and return a list of results
    results = []
    results.append(check_accuracy(df))
    results.append(check_completeness(df))
    results.append(check_consistency(df))
    results.append(check_timeliness(df))
    results.append(check_relevance(df))
    results.append(check_validity(df))
    results.append(check_precision(df))
    results.append(check_uniqueness(df))
    return results

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render home page template
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/data_quality_checks/")
async def data_quality_checks(request: Request, file: bytes = Form(...)):
    # Load data and run data quality checks
    df = pd.read_csv(file)
    results = run_data_quality_checks(df)
    
    # Prepare response
    response = []
    for i, result in enumerate(results):
        response.append({'check_number': i+1, 'result': 'Passed' if result else 'Failed'})
    
    # Render results page template
    return templates.TemplateResponse("results.html", {"request": request, "results": response})

