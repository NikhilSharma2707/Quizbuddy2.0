import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import logging
from typing import List, Dict, Optional
import random
import sys
import traceback
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quiz_generator.log')
    ]
)

logger = logging.getLogger(__name__)


class BartQuizGenerator:
    def __init__(self, model_name: str = "facebook/bart-large", device: str = None):
        """
        Initialize the quiz generator using BART model.
        Args:
            model_name: The BART model to use (default: facebook/bart-large)
            device: Device to run the model on (cuda/cpu)
        """
        try:
            logger.info("Initializing BartQuizGenerator...")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.device = device
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

            # Specific prefixes for question generation
            self.difficulty_prefixes = {
                "easy": "basic: ",
                "medium": "intermediate: ",
                "hard": "advanced: "
            }

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _prepare_input_text(self, context: str, difficulty: str = "medium") -> str:
        """Prepare input text with the appropriate prefix and format."""
        prefix = self.difficulty_prefixes.get(difficulty, self.difficulty_prefixes["medium"])
        context = context.strip().replace("\n", " ")
        return f"Generate a question: {prefix}{context}"

    def generate_question(
            self,
            context: str,
            difficulty: str = "medium",
            num_questions: int = 1,
            max_length: int = 64,
            min_length: int = 10,
            temperature: float = 0.8
    ) -> List[str]:
        """
        Generate questions using BART model.

        Args:
            context: Text to generate questions from
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_questions: Number of questions to generate
            max_length: Maximum length of generated questions
            min_length: Minimum length of generated questions
            temperature: Controls randomness in generation

        Returns:
            List of generated questions
        """
        try:
            input_text = self._prepare_input_text(context, difficulty)

            encoding = self.tokenizer(
                input_text,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate questions using BART
            outputs = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_questions,
                num_beams=4,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                do_sample=True,
                early_stopping=True
            )

            questions = []
            for output in outputs:
                question = self.tokenizer.decode(output, skip_special_tokens=True)
                question = question.replace("Generate a question:", "").strip()
                questions.append(question)

            return questions

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def generate_quiz_with_options(self, context: str, difficulty: str = "medium") -> Dict:
        """
        Generate a complete quiz question with options using BART.

        Args:
            context: The context to generate the question from
            difficulty: Difficulty level of the question

        Returns:
            Dictionary containing question, correct answer, and options
        """
        try:
            # Generate question
            questions = self.generate_question(context, difficulty)
            if not questions:
                raise ValueError("Failed to generate question")

            question = questions[0]

            # Generate answer
            answer_prompt = f"Answer this question: {question}\nContext: {context}"
            answer_encoding = self.tokenizer(
                answer_prompt,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            answer_output = self.model.generate(
                input_ids=answer_encoding.input_ids,
                attention_mask=answer_encoding.attention_mask,
                max_length=32,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )

            correct_answer = self.tokenizer.decode(answer_output[0], skip_special_tokens=True)
            correct_answer = correct_answer.replace("Answer:", "").strip()

            # Generate distractors (wrong options)
            options = [correct_answer]
            for _ in range(3):  # Generate 3 wrong options
                distractor_prompt = f"Generate a wrong but plausible answer: {question}\nContext: {context}"
                distractor_encoding = self.tokenizer(
                    distractor_prompt,
                    max_length=1024,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                distractor_output = self.model.generate(
                    input_ids=distractor_encoding.input_ids,
                    attention_mask=distractor_encoding.attention_mask,
                    max_length=32,
                    num_beams=4,
                    temperature=0.9,
                    do_sample=True
                )

                distractor = self.tokenizer.decode(distractor_output[0], skip_special_tokens=True)
                distractor = distractor.replace("Generate a wrong but plausible answer:", "").strip()

                if distractor not in options:
                    options.append(distractor)

            # Ensure we have enough options
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")

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
            logger.error(traceback.format_exc())
            return None


def main():
    print("=== BART Quiz Generator Test ===")
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

    try:
        quiz_gen = BartQuizGenerator()

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

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        print("An error occurred during testing. Check quiz_generator.log for details.")


if __name__ == "__main__":
    main()