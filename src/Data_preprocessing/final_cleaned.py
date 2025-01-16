import pandas as pd
import re
import os
import warnings
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download('punkt', quiet=True)


def estimate_difficulty(text):
    # Count words
    word_count = len(word_tokenize(text))

    # Count complex words (words with 3 or more syllables)
    complex_words = len([word for word in word_tokenize(text) if count_syllables(word) >= 3])

    # Calculate complexity score
    complexity_score = (word_count + complex_words) / 2

    # Classify difficulty based on complexity score
    if complexity_score < 10:
        return 'easy'
    elif complexity_score < 20:
        return 'medium'
    else:
        return 'hard'


def count_syllables(word):
    # Simple syllable counting heuristic
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count += 1
    return count


def preprocess_dataset(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    print("Processing the dataset...")

    # Clean and process the input
    def process_input(text):
        # Remove 'generate question:' prefix
        text = re.sub(r'^generate question:\s*', '', text, flags=re.IGNORECASE).strip()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    df['processed_input'] = df['input'].apply(process_input)

    # Estimate difficulty
    df['difficulty'] = df['processed_input'].apply(estimate_difficulty)

    # Simple token length check (approximate)
    df['token_length'] = df['processed_input'].apply(lambda x: len(x.split()))
    df = df[df['token_length'] <= 512]

    # Remove duplicates
    df.drop_duplicates(subset=['processed_input'], keep='first', inplace=True)

    # Save the preprocessed dataset
    df.to_csv(output_file, index=False)

    print(f"Preprocessed dataset saved to {output_file}")
    print(f"Final dataset size: {len(df)}")
    print(f"Average token length: {df['token_length'].mean():.2f}")
    print(f"Max token length: {df['token_length'].max()}")
    print("Distribution of difficulties:")
    print(Counter(df['difficulty']))


if __name__ == "__main__":
    # Use raw string for Windows paths
    input_file = r"C:\quizbuddy2.0\src\Data_preprocessing\t5_ready_data.csv"
    output_directory = r"C:\quizbuddy2.0\src\Data_collection"
    output_filename = "preprocessed_dataset.csv"

    # Create the full output file path
    output_file = os.path.join(output_directory, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
    else:
        preprocess_dataset(input_file, output_file)