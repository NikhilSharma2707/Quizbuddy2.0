from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pypdf import PdfReader
import torch
import json
from typing import List, Dict


class QuizBuddy:
    def __init__(self):
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

    def extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages)

    def generate_question(self, context: str, difficulty: float = 0.5) -> Dict:
        prompt = f"""Generate a multiple-choice question from this text: "{context}"
        Difficulty level: {difficulty} (0-1)
        Format: JSON with keys: question, options (4 choices), correct_index (0-3), explanation"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs.to(self.device),
            max_length=800,
            temperature=0.7,
            top_p=0.9
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # Extract JSON from response
            json_str = response[response.find("{"):response.rfind("}") + 1]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def adapt_difficulty(self, user_performance: List[Dict]) -> float:
        """Adjust difficulty based on user's performance history"""
        if not user_performance:
            return 0.5

        recent_scores = [entry["correct"] for entry in user_performance[-5:]]
        accuracy = sum(recent_scores) / len(recent_scores)

        if accuracy > 0.8:
            return min(1.0, accuracy + 0.1)
        elif accuracy < 0.4:
            return max(0.1, accuracy - 0.1)
        return accuracy

    def generate_quiz(self, pdf_path: str, num_questions: int = 5) -> List[Dict]:
        """Generate a complete quiz from PDF content"""
        content = self.extract_pdf_content(pdf_path)
        questions = []

        for _ in range(num_questions):
            question = self.generate_question(content)
            if question:
                questions.append(question)

        return questions


# Example usage:
if __name__ == "__main__":
    quiz = QuizBuddy()
    # Generate quiz from PDF
    questions = quiz.generate_quiz("study_material.pdf", num_questions=5)
    print(json.dumps(questions, indent=2))