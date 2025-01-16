import pandas as pd
import re
import os

# Define file paths
input_file = r"C:\quizbuddy2.0\src\Data_collection\openstax_books.csv"
output_file = r"C:\quizbuddy2.0\src\Data_collection\processed_openstax_books.csv"


def clean_description(text):
    if pd.isna(text):
        return ''
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text.strip()


def generate_questions(title, description):
    questions = [
        f"What topics are covered in '{title}'?",
        f"How does '{title}' approach {description.split('.')[0]}?",
        f"What are the key concepts in '{title}'?"
    ]
    return questions


def process_data(input_file, output_file):
    # Check if processed file already exists
    if os.path.exists(output_file):
        print(f"Processed file already exists: {output_file}")
        return pd.read_csv(output_file)

    # Load the CSV file
    df = pd.read_csv(input_file)

    # Apply the cleaning function
    df['cleaned_description'] = df['description'].apply(clean_description)

    # Generate questions
    df['generated_questions'] = df.apply(lambda x: generate_questions(x['title'], x['cleaned_description']), axis=1)

    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")

    return df


# Process the data
processed_df = process_data(input_file, output_file)

# Display the updated DataFrame
print(processed_df[['title', 'cleaned_description', 'generated_questions']])