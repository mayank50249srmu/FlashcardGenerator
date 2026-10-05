import os
import json

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


def generate_flashcards(notes):
    # Read your Hugging Face token from the .env file
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("Add your HF_TOKEN to a .env file first.")

    # Connect to Hugging Face
    client = InferenceClient(api_key=token)

    prompt = f"""
Make 4 to 8 study flashcards from these notes.
Return only valid JSON in this format:
{{
  "cards": [
    {{"question": "A question", "answer": "The answer"}}
  ]
}}

Notes:
{notes}
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    # Get the text returned by the model
    text = response.choices[0].message.content.strip()

    # Remove code fences if the model included them
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    # Turn the JSON text into a Python dictionary
    data = json.loads(text)

    return data["cards"]


def study_session(cards):
    print(f"\nStudy session: {len(cards)} cards")

    for number, card in enumerate(cards, start=1):
        print(f"\nCard {number}")
        print("Q:", card["question"])
        print("A:", card["answer"])

        if number < len(cards):
            choice = input("\nPress Enter for the next card, or type q to quit: ")
            if choice.lower() == "q":
                print("Session ended.")
                return

    print("\nYou finished all the cards!")


if __name__ == "__main__":
    notes = """
    Polymorphism lets methods behave differently depending on the object.

    Duck typing means Python checks whether an object has the needed methods
    or attributes instead of checking its exact type.

    Method overriding happens when a subclass replaces a method from its
    superclass.

    Python does not support traditional method overloading by default, but
    default arguments or *args can be used for similar behavior.
    """

    print("Creating flashcards...")
    flashcards = generate_flashcards(notes)
    study_session(flashcards)