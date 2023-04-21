import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

# Load the trained models and scaler
knn_model = load('knn_model.joblib')
rfc_model = load('rfc_model.joblib')
svc_model = load('svc_model.joblib')
lr_model = load('lr_model.joblib')
scaler = load('scaler.joblib')

# Define the FastAPI app and input data model
app = FastAPI(title = "Iris Classification", description = "Machine Learning Models: KNN, Random Forest Classifier,SVM Classifer, and Logistic Regression")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# create a dictionary to map the predicted class labels to string flower names
class_label_to_name = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
    }

@app.get("/")
async def read():
    return {"message": "Choose a Machine Learning Model and input the flower features for a species prediction."}

@app.post("/knn")
async def knn_predict(data: IrisData):
    # Scale the input data using the loaded StandardScaler
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained KNN model
    prediction = knn_model.predict(input_data_scaled)
    
    # map the predicted class label to a string flower name
    flower_name = class_label_to_name[prediction[0]]

    # return the predicted flower name as a JSON response
    return {"flower_name_prediction": flower_name}


@app.post("/rfc")
async def rfc_predict(data: IrisData):
    # Scale the input data using the loaded StandardScaler
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained Random Forest Classifier model
    prediction = rfc_model.predict(input_data_scaled)

    # map the predicted class label to a string flower name
    flower_name = class_label_to_name[prediction[0]]

    # return the predicted flower name as a JSON response
    return {"flower_name_prediction": flower_name}

@app.post("/svc")
async def svc_prediction(data: IrisData):
    # Scale the input data using the loaded StandardScaler
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained SVM Classifier model
    prediction = svc_model.predict(input_data_scaled)

    # map the predicted class label to a string flower name
    flower_name = class_label_to_name[prediction[0]]

    # return the predicted flower name as a JSON response
    return {"flower_name_prediction": flower_name}

@app.post("/lr")
async def lr_prediction(data: IrisData):
    # Scale the input data using the loaded StandardScaler
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using the trained Logistic Regression Classifier model
    prediction = lr_model.predict(input_data_scaled)

    # map the predicted class label to a string flower name
    flower_name = class_label_to_name[prediction[0]]

    # return the predicted flower name as a JSON response
    return {"flower_name_prediction": flower_name}