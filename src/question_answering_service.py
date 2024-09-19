import requests

class QuestionAnsweringService:
    def __init__(self, model_name='phi3:mini'):
        self.model_name = model_name  # Ollama model name (e.g., 'llama')

    def generate_answer(self, question, context):
        # Combine the question and context into a prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

        # Make a request to the Ollama API with 'stream' set to False
        response = requests.post(
            'http://localhost:11434/api/generate',  # Ollama API running locally
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False  # Disable streaming
            }
        )

        # Print the raw response for debugging
        # print("Raw response from Ollama API:", response.json()['response'])

        # Check if the response is valid and process it
        if response.status_code == 200:
            try:
                data = response.json()  # Parse the JSON response
                return data.get("response", "No answer found.")
            except requests.exceptions.JSONDecodeError:
                return f"Error: Could not parse JSON response. Raw response: {response.json()['response']}"
        else:
            return f"Error: {response.status_code} - {response.json()['response']}"

