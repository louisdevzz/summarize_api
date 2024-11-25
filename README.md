## Vietnamese Text Summarizer API

This API provides a simple interface for summarizing Vietnamese text using a transformer model.

### API Endpoints

- `POST /summarize`: Summarize a given text.

### Example Usage

You can test the API using the following Python code:

#### Run the API
```bash
python main.py
#or
python3 main.py

```

#### Test the API
create a file test_api.py and run:
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

