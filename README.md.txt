# My AI/ML Assignment Submission

## Overview
This repository contains the solution to the AI/ML assignment, covering various tasks in Machine Learning, Deep Learning, and Natural Language Processing. The project utilizes popular Python libraries such as Scikit-learn, TensorFlow/Keras, and spaCy. It also includes a bonus task for deploying a Deep Learning model using Streamlit.

## Project Structure
* `My_ML_Assignment.ipynb`: The main Google Colab notebook containing all the code and explanations for Task 1, Task 2, and Task 3.
* `app.py`: The Streamlit application script for the Bonus Task, used to deploy the MNIST classifier.
* `mnist_cnn_model.h5`: The pre-trained Convolutional Neural Network (CNN) model for MNIST digit classification, saved from Task 2.
* `README.md`: This file, providing an overview of the repository.

## Task Breakdown

### Task 1: Classical ML with Scikit-learn
* **Dataset:** Iris Species Dataset.
* **Goal:** Preprocess data, train a Decision Tree Classifier, and evaluate its performance (accuracy, precision, recall).

### Task 2: Deep Learning with TensorFlow/Keras
* **Dataset:** MNIST Handwritten Digits Dataset.
* **Goal:** Build and train a Convolutional Neural Network (CNN) to achieve >95% test accuracy for digit classification. Includes model evaluation and visualization of predictions. The trained model is saved as `mnist_cnn_model.h5`.

### Task 3: NLP with spaCy
* **Text Data:** Sample Amazon Product Reviews.
* **Goal:** Perform Named Entity Recognition (NER) to extract product names and brands, and analyze sentiment using a rule-based approach. Includes visualization of entities and sentiment distribution.

### Bonus Task: Model Deployment with Streamlit
* **Model:** MNIST Digit Classifier (from Task 2).
* **Deployment:** A Streamlit web application (`app.py`) is provided, allowing users to upload an image of a handwritten digit and get a prediction from the trained CNN model.

## How to Run the Project

### Using Google Colab (Recommended)
1.  Upload `My_ML_Assignment.ipynb` to Google Colab.
2.  Run all cells in sequence.
3.  For the **Bonus Task (Streamlit Deployment)**:
    * Ensure `mnist_cnn_model.h5` is saved after running Task 2.
    * The `app.py` script is generated within the notebook using `%%writefile`.
    * Follow the instructions in the notebook cells to install `pyngrok` and obtain an `ngrok` authentication token from [ngrok.com](https://ngrok.com).
    * Set your `ngrok` authtoken in the provided cell (`os.environ["NGROK_AUTH_TOKEN"] = "YOUR_TOKEN"`).
    * Run the Streamlit deployment cells; a public URL will be provided.

### Running Locally (Advanced - Requires Anaconda Environment Setup)
1.  **Clone this repository** to your local machine.
2.  **Create and activate a Conda environment** (e.g., `my_ml_env`) with Python 3.9:
    ```bash
    conda create -n my_ml_env python=3.9
    conda activate my_ml_env
    ```
3.  **Install all required libraries:**
    ```bash
    conda install jupyterlab pandas matplotlib scikit-learn tensorflow spacy -c conda-forge
    python -m spacy download en_core_web_sm
    pip install streamlit pillow # Streamlit is often easier with pip for specific features
    ```
4.  **Launch Jupyter Lab/Notebook:**
    ```bash
    jupyter lab
    ```
    Open `My_ML_Assignment.ipynb` and run the cells.
5.  **For Streamlit (`app.py`)**:
    * Navigate to the directory containing `app.py` in your terminal within the activated `my_ml_env`.
    * Run: `streamlit run app.py`
    * This will typically open the app in your local browser.

## Deliverables
* The `.ipynb` notebook with all code and outputs.
* The `app.py` Streamlit script.
* The `mnist_cnn_model.h5` trained model file.
* A screenshot of the running Streamlit application.
* A live demo URL for the Streamlit application (temporary if using free `ngrok` in Colab).

---