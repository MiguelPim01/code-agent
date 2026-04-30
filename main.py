import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Chatbot")
    
    parser.add_argument("user_prompt", type=str, help="User prompt")
    
    args = parser.parse_args()
    
    return args

def main():
    load_dotenv()
    
    # Parsing args
    args = parse_args()
    
    user_prompt = args.user_prompt
    
    # Creating Google API client
    api_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    
    # Storing a history of messages
    messages = [types.Content(role="user", parts=types.Part(text=user_prompt))]
    
    # Prompting the model
    response = client.models.generate_content(model="gemini-2.5-flash", contents=user_prompt)
    
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print(f"Model response: {response.text}")


if __name__ == "__main__":
    main()
