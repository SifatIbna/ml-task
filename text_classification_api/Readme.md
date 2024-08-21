# Sentiment Analysis API

This project implements a REST API for sentiment analysis using FastAPI and a pre-trained model from Hugging Face's transformers library.

## Overview

The API uses the "finiteautomata/bertweet-base-sentiment-analysis" model, which is specifically trained for sentiment analysis on Twitter data. It classifies text into sentiment categories.

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/SifatIbna/ml-task.git
   cd text_classification_api
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv # On linux, use, `python3 -m venv venv`
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install fastapi uvicorn transformers torch
   pip freeze > requirements.txt
   ```

## Running the API

To run the API locally:

```
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the API is running, you can access the Swagger UI documentation at `http://localhost:8000/docs`.

## API Endpoints

### POST /classify

Classifies the sentiment of the provided text.

Request body:
```json
{
  "text": "Your text here"
}
```

Response:
```json
{
  "class": "POS",
  "confidence": 0.9998
}
```

The exact labels returned depend on the model's classification scheme.

## Model Details

This API uses the "finiteautomata/bertweet-base-sentiment-analysis" pre-trained model from Hugging Face's `transformers` library. This model is fine-tuned for sentiment analysis on Twitter data, making it particularly suitable for short, informal text classification.

## Error Handling

The API implements basic error handling. If an exception occurs during classification, it will return a 500 Internal Server Error with details about the exception.

## Future Improvements

1. Add more comprehensive error handling and logging.
2. Implement authentication and rate limiting for API security.
3. Add caching to improve performance for repeated queries.
4. Expand the API to return multiple sentiment probabilities instead of just the top one.
5. Include example usage and sample calls in the documentation.

## Contributing

Contributions to improve the API are welcome. Please feel free to submit a Pull Request.

## License

[Include your chosen license here]

