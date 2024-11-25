## Vietnamese Text Summarizer API

A FastAPI-based REST API that provides text summarization capabilities for Vietnamese text using transformer models. The API uses a fine-tuned T5 model specifically trained for Vietnamese text summarization.

## Features

- Text summarization for Vietnamese content
- Configurable summary length
- REST API interface
- Automatic model training if no pre-trained model exists
- Easy to deploy and scale

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- 4GB+ RAM
- CUDA 11.8 (for GPU support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/louisdevzz/summarize_api.git
cd summarize_api
```

2. Install PyTorch with CUDA support:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install other dependencies:
```bash
pip3 install -r requirements.txt
```

4. Prepare your training data:
   - Place your `vietnamese_articles.csv` file in the root directory
   - The CSV should have 'text' and 'summary' columns

## Running the API

1. Start the API server:
```bash
python3 main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### POST /summarize

Summarizes the provided Vietnamese text.

#### Request Body

```json
{
    "text": "Your Vietnamese text to summarize",
    "max_length": 1000,  // Optional: maximum length of summary
    "min_length": 100    // Optional: minimum length of summary
}
```

#### Response

```json
{
    "original_length": 1500,
    "summary": "Summarized text",
    "summary_length": 500
}
```

## Testing the API

You can test the API using the provided test script:

```python
import requests

url = "http://localhost:8000/summarize"
payload = {
    "text": "Your input text here",
    "max_length": 1000,
    "min_length": 100
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

## Model Training

The system will automatically train a new model if no pre-trained model exists at `./saved_models/vietnamese_summarizer.pkl`. To force retraining:

1. Delete or rename the existing model file
2. Restart the API server

## Project Structure

```
vietnamese-summarizer/
├── main.py              # FastAPI application
├── model.py             # Summarizer model implementation
├── requirements.txt     # Project dependencies
├── README.md           # Documentation
└── saved_models/       # Directory for saved models
```

## Error Handling

The API includes proper error handling for:
- Invalid input text
- Model loading failures
- Training data issues
- Runtime errors

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the T5 Vietnamese summarization model by [pengold](https://huggingface.co/pengold/t5-vietnamese-summarization)
- FastAPI framework
- Hugging Face Transformers library

