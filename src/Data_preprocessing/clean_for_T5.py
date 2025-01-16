import pandas as pd
import re
from bs4 import BeautifulSoup


def clean_text(text):
    # Check if the input is not a string (e.g., NaN or float)
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Replace \xa0 with space
    text = text.replace(u'\xa0', u' ')
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_for_t5(df):
    # Clean title and description
    df['title'] = df['title'].apply(clean_text)
    df['cleaned_description'] = df['cleaned_description'].apply(clean_text)

    # Explode the generated questions
    df = df.explode('generated_questions')

    # Create input-output pairs
    df['input'] = 'generate question: ' + df['cleaned_description']
    df['output'] = df['generated_questions']

    # Truncate long texts (adjust max_length as needed)
    max_length = 512
    df['input'] = df['input'].apply(lambda x: x[:max_length])

    return df[['input', 'output']]


# Load your data
df = pd.read_csv("C:\quizbuddy2.0\src\Data_collection\processed_openstax_books.csv")

# Convert string representation of list to actual list
df['generated_questions'] = df['generated_questions'].apply(eval)

# Preprocess
preprocessed_df = preprocess_for_t5(df)

# Save preprocessed data
preprocessed_df.to_csv('t5_ready_data.csv', index=False)

print(preprocessed_df.head())