import openai
import logging
import random
import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quiz_generator_gpt3.log')
    ]
)

logger = logging.getLogger(__name__)

class GPT3QuizGenerator:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the quiz generator using GPT-3.5 model.
        Args:
            api_key: API key for OpenAI
            model_name: The GPT-3 model to use (default: gpt-3.5-turbo)
        """
        try:
            logger.info("Initializing GPT3QuizGenerator...")
            openai.api_key = api_key
            self.model_name = model_name

            # Difficulty prefixes for prompt customization
            self.difficulty_prefixes = {
                "easy": "Create a basic question: ",
                "medium": "Create an intermediate question: ",
                "hard": "Create an advanced question: "
            }

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def _prepare_prompt(self, context: str, difficulty: str = "medium") -> str:
        """Prepare prompt text with appropriate difficulty level."""
        prefix = self.difficulty_prefixes.get(difficulty, self.difficulty_prefixes["medium"])
        context = context.strip().replace("\n", " ")
        return f"{prefix}{context}"

    def generate_question(self, context: str, difficulty: str = "medium") -> str:
        """
        Generate a question using GPT-3.

        Args:
            context: Text to generate questions from
            difficulty: Difficulty level ('easy', 'medium', 'hard')

        Returns:
            A generated question
        """
        try:
            prompt = self._prepare_prompt(context, difficulty)

            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )

            if response and response.choices:
                question = response.choices[0].text.strip()
                return question
            else:
                logger.error("Empty response or no choices found in GPT-3 response.")
                return ""

        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return ""

    def generate_quiz_with_options(self, context: str, difficulty: str = "medium") -> Dict:
        """
        Generate a complete quiz question with options using GPT-3.

        Args:
            context: The context to generate the question from
            difficulty: Difficulty level of the question

        Returns:
            Dictionary containing question, correct answer, and options
        """
        try:
            # Generate question
            question = self.generate_question(context, difficulty)
            if not question:
                raise ValueError("Failed to generate question")

            # Generate correct answer
            answer_prompt = f"Provide the correct answer to the following question: {question}\nContext: {context}"
            answer_response = openai.Completion.create(
                model=self.model_name,
                prompt=answer_prompt,
                max_tokens=20,
                temperature=0.5
            )
            if answer_response and answer_response.choices:
                correct_answer = answer_response.choices[0].text.strip()
            else:
                logger.error("Empty response or no choices found for correct answer.")
                return None

            # Generate distractor options
            options = [correct_answer]
            for _ in range(3):  # Generate 3 wrong options
                distractor_prompt = f"Provide a plausible but incorrect answer for the question: {question}\nContext: {context}"
                distractor_response = openai.Completion.create(
                    model=self.model_name,
                    prompt=distractor_prompt,
                    max_tokens=20,
                    temperature=0.9
                )
                if distractor_response and distractor_response.choices:
                    distractor = distractor_response.choices[0].text.strip()
                    if distractor not in options:
                        options.append(distractor)
                else:
                    logger.error("Empty response or no choices found for distractor.")
                    continue

            random.shuffle(options)

            return {
                "question": question,
                "correct_answer": correct_answer,
                "options": options,
                "context": context,
                "difficulty": difficulty
            }

        except Exception as e:
            logger.error(f"Error generating quiz with options: {str(e)}")
            return None

def main():
    api_key = os.getenv("OPENAI_API_KEY")  # Fetch the key from environment
    print("=== GPT-3 Quiz Generator Test ===")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Test contexts
    contexts = [
        """
        The water cycle, also known as the hydrologic cycle, describes the continuous 
        movement of water within the Earth and atmosphere. It involves processes such 
        as evaporation, condensation, precipitation, and runoff.
        """,
        """
        The solar system consists of the Sun and all celestial objects bound to it by gravity, 
        including planets, dwarf planets, and numerous smaller bodies like asteroids and comets. 
        Mercury is the closest planet to the Sun, while Neptune is currently the farthest planet.
        """
    ]

    quiz_gen = GPT3QuizGenerator(api_key)

    for i, context in enumerate(contexts, 1):
        print(f"\n=== Test {i} ===")
        print(f"Context: {context.strip()}\n")

        for difficulty in ['easy', 'medium', 'hard']:
            print(f"\nDifficulty: {difficulty}")
            result = quiz_gen.generate_quiz_with_options(context, difficulty)

            if result:
                print("Question:", result['question'])
                print("\nOptions:")
                for j, option in enumerate(result['options'], 1):
                    is_correct = option == result['correct_answer']
                    print(f"{j}. {option}{' (Correct)' if is_correct else ''}")
            else:
                print(f"Failed to generate {difficulty} question for this context.")

if __name__ == "__main__":
    main()