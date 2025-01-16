
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional


class OptimizedQuizGenerator:
    def __init__(self, model_name: str = "mrm8488/t5-base-finetuned-question-generation-ap", device: str = None):
        """
        Initialize the quiz generator using a model fine-tuned for question generation.
        Uses a T5 model specifically fine-tuned for question generation tasks.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Specific prefixes for question generation
        self.difficulty_prefixes = {
            "easy": "basic: ",
            "medium": "intermediate: ",
            "hard": "advanced: "
        }

    def _prepare_input_text(self, context: str, difficulty: str) -> str:
        """Prepare input text in the format expected by the question generation model."""
        prefix = self.difficulty_prefixes.get(difficulty, self.difficulty_prefixes["medium"])
        # Clean the context
        context = context.strip().replace("\n", " ")
        # Format: "generate question: {context}"
        return f"generate question: {prefix}{context}"

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
        Generate questions based on the given context and difficulty level.

        Args:
            context: Text to generate questions from
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            num_questions: Number of questions to generate
            max_length: Maximum length of generated questions
            min_length: Minimum length of generated questions
            temperature: Controls randomness (higher = more random)

        Returns:
            List of generated questions
        """
        input_text = self._prepare_input_text(context, difficulty)

        # Encode the input text
        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        # Generate questions
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

        # Decode and clean up the generated questions
        questions = []
        for output in outputs:
            question = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove any "generate question:" prefix if it appears in the output
            question = question.replace("generate question:", "").strip()
            questions.append(question)

        return questions

    def batch_generate_questions(
            self,
            contexts: List[str],
            difficulties: Optional[List[str]] = None,
            questions_per_context: int = 1
    ) -> Dict[str, List[str]]:
        """Generate questions for multiple contexts in batch."""
        if difficulties is None:
            difficulties = ["medium"] * len(contexts)

        if len(contexts) != len(difficulties):
            raise ValueError("Number of contexts must match number of difficulties")

        results = {}
        for context, difficulty in zip(contexts, difficulties):
            questions = self.generate_question(
                context=context,
                difficulty=difficulty,
                num_questions=questions_per_context
            )
            results[context] = questions

        return results


def main():
    print("Initializing Quiz Generator...")
    quiz_gen = OptimizedQuizGenerator()

    # Test with water cycle context
    context = """
    The water cycle, also known as the hydrologic cycle, describes the continuous 
    movement of water within the Earth and atmosphere. It involves processes such 
    as evaporation, condensation, precipitation, and runoff.
    """

    print("\nGenerating questions for different difficulty levels...")
    for difficulty in ['easy', 'medium', 'hard']:
        questions = quiz_gen.generate_question(
            context=context,
            difficulty=difficulty,
            num_questions=1,
            temperature=0.8
        )
        print(f"\n{difficulty.upper()} question:")
        print(questions[0])

    # Test batch generation
    print("\nTesting batch generation...")
    contexts = [
        "Photosynthesis is the process by which plants convert sunlight into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
        "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries, marked by renewed interest in classical art and learning."
    ]
    difficulties = ['easy', 'hard']

    print("\nGenerating batch questions...")
    batch_results = quiz_gen.batch_generate_questions(
        contexts=contexts,
        difficulties=difficulties,
        questions_per_context=1
    )

    for context, questions in batch_results.items():
        print(f"\nContext: {context[:50]}...")
        for q in questions:
            print(f"Generated Question: {q}")


if __name__ == "__main__":
    main()