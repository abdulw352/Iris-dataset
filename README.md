# Iris-dataset 💐

# Iris Classification Models
This project uses the Iris dataset to train three different classification models: KNN, Random Forest Classifier, SVM Classifier, and Logistic Regression Classifier. It also includes a FastAPI implementation that exposes endpoints to execute each of the trained models.

 ### Objective 
 The objective of this project is to develop a classification model that can accurately predict the type of iris flower based on the given attributes, such as sepal length, sepal width, petal length, and petal width. The purpose of this project is to showcase how to train different classification models on a small dataset and deploy them using FastAPI.

### Dataset
The Iris dataset consists of 150 instances of iris flowers, with 50 instances for each of the three different types of iris flowers (Setosa, Versicolor, and Virginica). Each instance includes four attributes: sepal length, sepal width, petal length, and petal width.

### Instructions for running the FASTAPI project
1. Clone the repository
    1. Open a command prompt or terminal window and navigate to the directory where you want to download the project.
    1. run the following command to clone the repository:
    ```https://github.com/abdulw352/Iris-dataset```

1. Install dependencies:
    1. Navigate into the project directory that you just cloned and install the required dependencies by running the following command:
    ```pip install -r requirements.txt```

1. Run the server:
    1. Once the dependencies are installed, you can start the FastAPI server by running the following command in the project directory:
    ```uvicorn main:app --reload```

    1. This will start the server on the default port 8000. If you want to use a different port, you can add the --port option followed by the desired port number.

1. Test the API:
    1. Open a web browser and navigate to http://localhost:8000/docs to access the automatically generated API documentation. From here, you can test the endpoints and methods of the API.



