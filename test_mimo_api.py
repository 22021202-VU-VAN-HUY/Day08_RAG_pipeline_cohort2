import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

mimo_key = os.getenv("MIMO_API_KEY")
mimo_url = os.getenv("MIMO_BASE_URL")
mimo_model = os.getenv("MIMO_MODEL")

print(f"API Key: {mimo_key[:20]}...")
print(f"Base URL: {mimo_url}")
print(f"Model: {mimo_model}")
print()

try:
    client = OpenAI(api_key=mimo_key, base_url=mimo_url)
    print("Creating chat completion...")
    response = client.chat.completions.create(
        model=mimo_model,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )
    print("✅ Success!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()
