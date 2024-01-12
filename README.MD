# Paraphrase Detection - NLP Course Assignment

## created by : 
- Kaleab Taye - UGR/0490/12
- Estifanos Neway - UGR/4776/12
- Beka Dessalegn - UGR/4605/12

## stream : 
- software

## Description
In this project we have attempted to build and train a model that determine whether two given text entities, such as sentences, convey the same meaning in different words. In order to obtain the paraphrase detection capablity we have tried to implement Bidirectional Long Short-Term Memory (Bi-LSTM) algorithm. The model is designed to analyze both syntactic and semantic features of input text pairs, providing a robust solution to the paraphrase detection challenge.

## Usage Guideline
- the training process can be found in paraphrase-detection-nlp/Model and Server/model/model.ipynb
- to start the back-end 
    - install the required fast api packages ```pip install fastapi uvicorn starlette pydantic``` 
    - to start the application go to /server directory and run the command ```uvicorn paraphrase_detection:app --reload```
    - now send post request to the printed_url/checkparaphrase with the message body containing two parameters
        - sentence1: the first sentence 
        - sentence2: the second sentence
    - the response will be a json object containing the following keys: 
        - paraphrase_result: whether the two sentences are paraphrase or not
        - parphrase_probability: the score of the model for the two sentences
- to start the front-end
    - install the required packages ```npm install```
    - run the command ```npm run dev```
    - navigate to http://localhost:3000/