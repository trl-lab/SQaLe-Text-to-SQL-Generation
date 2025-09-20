import sys
from abc import ABC, abstractmethod
from ReFoRCE.utils import extract_all_blocks

class BaseChat(ABC):
    def __init__(self, model: str, temperature: float = 1.0):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []

    @abstractmethod
    def get_response(self, prompt) -> str:
        pass

    def get_model_response(self, prompt, code_format=None) -> list:
        code_blocks = []
        max_try = 5
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
            code_blocks = extract_all_blocks(response, code_format)
        if max_try == 0 or code_blocks == []:
            print(f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}")
            sys.exit(0)
        return code_blocks

    def get_model_response_txt(self, prompt) -> str:
        max_try = 3
        while max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
                return response
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
        print(f"get_model_response_txt() exit, max_try: {max_try}")
        sys.exit(0)

    def get_message_len(self):
        return {
            "prompt_len": sum(len(item["content"]) for item in self.messages if item["role"] == "user"),
            "response_len": sum(len(item["content"]) for item in self.messages if item["role"] == "assistant"),
            "num_calls": len(self.messages) // 2
        }

    def init_messages(self):
        self.messages = []


from openai import OpenAI, AzureOpenAI
import os

class GPTChat(BaseChat):
    def __init__(self, azure=False, model="gpt-4o", temperature=1.0):
        super().__init__(model, temperature)

        if not azure:
            if model in ["o1-preview", "o1-mini"]:
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    api_version="2024-12-01-preview"
                )
            elif model in ["deepseek-reasoner"]:
                self.client = OpenAI(
                    base_url="https://api.deepseek.com",
                    api_key=os.environ.get("DS_API_KEY"),
                )
            else:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            if model in ["o1-preview", "o1-mini", "o3", "o4-mini"]:
                version = "2024-12-01-preview"
            elif model in ["o3-pro"]:
                version = "2025-03-01-preview"
            else:
                version = "2024-05-01-preview"

            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_KEY"),
                api_version=version
            )

    def get_response(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})
        if self.model == "o3-pro":
            response = self.client.responses.create(
                model=self.model,
                input=self.messages,
                temperature=self.temperature
            )
            main_content = response.output_text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature
            )
            main_content = response.choices[0].message.content

        self.messages.append({"role": "assistant", "content": main_content})
        return main_content
    
from ollama import Client

class OllamaChat(BaseChat):
    def __init__(self, model: str = "llama3.1:8b-instruct", temperature: float = 0.7, host: str | None = None):
        super().__init__(model, temperature)
        # Allow override via arg or env; fall back to default local server
        self.client = Client(host=os.environ.get("OLLAMA_HOST", host or "http://localhost:11434"))

    def get_response(self, prompt) -> str:
        # Track the user message just like GPTChat
        self.messages.append({"role": "user", "content": prompt})
        try:
            # Ollama expects: [{"role": "...", "content": "..."}]
            # You can pass additional options here if you like (top_p, num_ctx, etc.)
            resp = self.client.chat(
                model=self.model,
                messages=self.messages,
                options={
                    "temperature": self.temperature
                },
            )
            main_content = resp["message"]["content"]
        except Exception as e:
            # Let BaseChat’s retry loop handle it
            raise e

        # Track assistant response
        self.messages.append({"role": "assistant", "content": main_content})
        return main_content