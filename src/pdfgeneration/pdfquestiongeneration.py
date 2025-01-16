import PyPDF2
from transformers import pipeline
from typing import List, Dict
import torch
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuizGenerator:
    def __init__(self):
        logger.info("Initializing Quiz Generator...")
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)

            logger.info("Loading question generation model...")
            self.question_generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",  # Smaller, free to use model
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Add QA model for better answer generation
            logger.info("Loading QA model...")
            self.answer_generator = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            logger.info("Models loaded successfully!")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return ""

    def split_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks for processing."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def generate_questions(self, text_chunk: str, num_questions: int = 3) -> List[Dict]:
        """Generate questions from a text chunk."""
        logger.info("Generating questions for chunk...")
        questions = []

        try:
            # Clean up text chunk
            text_chunk = text_chunk.replace('\n', ' ').strip()

            for _ in range(num_questions):
                # Generate question
                prompt = f"Generate a specific and clear question based on this text: {text_chunk}"
                response = self.question_generator(
                    prompt,
                    max_length=64,
                    num_return_sequences=1,
                    temperature=0.7  # Add some randomness
                )

                if response and len(response) > 0:
                    question = response[0]["generated_text"]

                    # Generate answer using QA model
                    answer = self.answer_generator(
                        question=question,
                        context=text_chunk
                    )

                    # Generate wrong options
                    options_prompt = f"Generate three incorrect but plausible answers for the question: {question}"
                    wrong_options = self.question_generator(
                        options_prompt,
                        max_length=128,
                        num_return_sequences=1
                    )[0]["generated_text"].split(',')[:3]

                    # Clean up options
                    options = [answer['answer']] + [opt.strip() for opt in wrong_options if opt.strip()]

                    question_dict = {
                        "question": question,
                        "context": text_chunk,
                        "options": options,
                        "correct_answer": answer['answer']
                    }
                    questions.append(question_dict)
                    logger.info(f"Generated question: {question}")

        except Exception as e:
            logger.error(f"Error generating questions: {e}")

        return questions

    def process_pdf(self, pdf_path: str, questions_per_chunk: int = 3) -> List[Dict]:
        """Process a PDF file and generate questions."""
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning("No text extracted from PDF")
            return []

        # Split into chunks
        chunks = self.split_into_chunks(text)

        # Generate questions for each chunk
        all_questions = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            chunk_questions = self.generate_questions(chunk, questions_per_chunk)
            all_questions.extend(chunk_questions)

        logger.info(f"Generated total of {len(all_questions)} questions")
        return all_questions


def main():
    logger.info("Starting quiz generation process...")

    try:
        quiz_gen = QuizGenerator()

        # Process a PDF and generate questions
        questions = quiz_gen.process_pdf("C:\\quizbuddy2.0\\pdfs\\Testcase_2.pdf")

        # Print generated questions
        if questions:
            for i, q in enumerate(questions, 1):
                print(f"\nQuestion {i}:")
                print(f"Q: {q['question']}")
                print("Options:")
                for j, opt in enumerate(q['options'], 1):
                    print(f"{j}. {opt}")
                print(f"Correct Answer: {q['correct_answer']}")
                print(f"Context: {q['context'][:100]}...")
        else:
            logger.warning("No questions were generated!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()