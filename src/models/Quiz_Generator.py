
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
from typing import List, Dict
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


class QuizGenerator:
    def __init__(self, model_name: str = "mrm8488/t5-base-finetuned-question-generation-ap", device: str = None):
        try:
            logger.info("Initializing QuizGenerator...")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.device = device
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_options(self, context: str, question: str, correct_answer: str, num_options: int = 3) -> List[str]:
        try:
            logger.info("Generating options...")
            options = [correct_answer]

            # Split context into sentences for better option generation
            sentences = context.split('.')
            key_terms = []
            for sentence in sentences:
                words = sentence.strip().split()
                key_terms.extend([w for w in words if len(w) > 4])

            for _ in range(num_options):
                # Generate plausible wrong answer
                input_text = f"answer incorrectly but plausibly: {question}\ncontext: {context}"

                encoding = self.tokenizer(
                    input_text,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model.generate(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    max_length=50,
                    num_beams=5,
                    temperature=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )

                option = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                option = option.strip()

                # Clean up the option
                option = option.replace("answer:", "").replace("question:", "").strip()

                # Validate the option
                if (option.lower() != correct_answer.lower() and
                        not any(opt.lower() == option.lower() for opt in options) and
                        len(option) > 3):
                    options.append(option)

            # If we don't have enough options, generate some generic ones
            while len(options) < num_options + 1:
                fake_option = f"None of the above {len(options)}"
                if fake_option not in options:
                    options.append(fake_option)

            # Shuffle the options
            random.shuffle(options)

            logger.info(f"Generated {len(options)} options successfully")
            return options
        except Exception as e:
            logger.error(f"Error generating options: {str(e)}")
            return [correct_answer] + [f"Option {i + 2}" for i in range(3)]

    def generate_question(self, context: str) -> Dict:
        try:
            logger.info("Generating question...")

            # Generate question
            input_text = f"generate question: {context}"
            encoding = self.tokenizer(
                input_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            question_output = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_length=64,
                num_beams=4,
                temperature=0.8,
                do_sample=True
            )

            question = self.tokenizer.decode(question_output[0], skip_special_tokens=True)
            question = question.replace("generate question:", "").replace("question:", "").strip()

            logger.info("Generated question successfully")

            # Generate answer
            logger.info("Generating answer...")
            answer_prompt = f"answer: {question}\ncontext: {context}"

            answer_encoding = self.tokenizer(
                answer_prompt,
                max_length=512,
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

            answer = self.tokenizer.decode(answer_output[0], skip_special_tokens=True)
            answer = answer.replace("answer:", "").replace("question:", "").strip()

            logger.info("Generated answer successfully")

            # Generate options
            options = self.generate_options(context, question, answer)

            return {
                "question": question,
                "correct_answer": answer,
                "options": options,
                "context": context
            }
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            logger.error(traceback.format_exc())
            return None


def test_quiz_generator():
    try:
        logger.info("Starting quiz generator test...")

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

        quiz_gen = QuizGenerator()

        for i, context in enumerate(contexts, 1):
            logger.info(f"\nTesting with context {i}...")
            print(f"\n=== Test {i} ===")
            print(f"Context: {context.strip()}\n")

            result = quiz_gen.generate_question(context)

            if result:
                print("Question:", result['question'])
                print("\nOptions:")
                for j, option in enumerate(result['options'], 1):
                    is_correct = option == result['correct_answer']
                    print(f"{j}. {option}{' (Correct)' if is_correct else ''}")
                print("\n")
            else:
                print("Failed to generate question for this context.")

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        print("An error occurred during testing. Check quiz_generator.log for details.")


if __name__ == "__main__":
    print("=== Quiz Generator Test ===")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    test_quiz_generator()