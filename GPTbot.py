import openai

# Initialize the API key
openai.api_key = "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"


# Generate a response using the ChatGPT model
def generate_response(prompt, model="gpt-3.5-turbo"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text.strip()
    return message

user_prompt = "Write a summary of the benefits of exercise."
chatbot_response = generate_response(user_prompt)
print(chatbot_response)
