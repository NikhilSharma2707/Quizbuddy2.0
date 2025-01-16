import logging
import random
import re
from datetime import datetime
from transformers import pipeline
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quiz_generator_gpt2.log')
    ]
)

logger = logging.getLogger(__name__)


class HuggingFaceQuizGenerator:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the quiz generator using a Hugging Face pre-trained model.
        Args:
            model_name: The Hugging Face model to use (default: "gpt2")
        """
        try:
            logger.info("Initializing HuggingFaceQuizGenerator...")
            self.model_name = model_name
            self.generator = pipeline("text-generation", model=self.model_name)

            # Difficulty prefixes for prompt customization
            self.difficulty_prefixes = {
                "easy": "Generate an easy question about the following context: ",
                "medium": "Generate a medium difficulty question about the following context: ",
                "hard": "Generate a challenging question about the following context: "
            }

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def _prepare_prompt(self, context: str, difficulty: str = "medium") -> str:
        """Prepare prompt text with appropriate difficulty level."""
        prefix = self.difficulty_prefixes.get(difficulty, self.difficulty_prefixes["medium"])
        context = context.strip().replace("\n", " ")
        return f"{prefix}{context}\n\nQuestion: "

    def generate_question(self, context: str, difficulty: str = "medium") -> str:
        """
        Generate a question using a Hugging Face pre-trained model.

        Args:
            context: Text to generate questions from
            difficulty: Difficulty level ('easy', 'medium', 'hard')

        Returns:
            A generated question
        """
        try:
            prompt = self._prepare_prompt(context, difficulty)
            response = self.generator(prompt, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.95,
                                      num_return_sequences=1, truncation=True)
            if response and response[0]['generated_text']:
                generated_text = response[0]['generated_text'].strip()

                # Extract the question from the generated text
                question_match = re.search(r'(?:^|\n)(.+\?)', generated_text)
                if question_match:
                    return question_match.group(1).strip()
                else:
                    # If no question mark is found, return the first sentence
                    sentences = re.split(r'(?<=[.!?])\s+', generated_text)
                    if sentences:
                        return sentences[0].strip()
                    else:
                        return "Failed to generate a clear question."
            else:
                logger.error("Empty response or no generated text found.")
                return "Failed to generate a question."

        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return "Error occurred while generating the question."

    def generate_quiz(self, context: str, difficulty: str = "medium") -> Dict:
        """
        Generate a complete quiz question using a Hugging Face pre-trained model.

        Args:
            context: The context to generate the question from
            difficulty: Difficulty level of the question

        Returns:
            Dictionary containing question
        """
        try:
            # Generate question
            question = self.generate_question(context, difficulty)
            if not question:
                raise ValueError("Failed to generate question")

            return {
                "question": question,
                "context": context,
                "difficulty": difficulty
            }

        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return None


def main():
    print("=== Hugging Face Quiz Generator Test ===")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Test contexts
    contexts = [
        """
        The Industrial Revolution was a period of significant economic, technological, and social changes that began in the late 18th century. It started in Britain and spread to other parts of the world. Major innovations like the steam engine, mechanized textile production, and the development of railroads marked this period.
        """,
        """
        Mount Everest, located in the Himalayas, is the highest mountain in the world, standing at 8,848 meters (29,029 feet) above sea level. It lies on the border between Nepal and the Tibet Autonomous Region of China.
        """
    ]

    quiz_gen = HuggingFaceQuizGenerator(model_name="gpt2")

    for i, context in enumerate(contexts, 1):
        print(f"\n=== Test {i} ===")
        print(f"Context: {context.strip()}\n")

        for difficulty in ['easy', 'medium', 'hard']:
            print(f"\nDifficulty: {difficulty}")
            result = quiz_gen.generate_quiz(context, difficulty)

            if result:
                print(f"Generated Question: {result['question']}")
            else:
                print(f"Failed to generate {difficulty} question for this context.")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()