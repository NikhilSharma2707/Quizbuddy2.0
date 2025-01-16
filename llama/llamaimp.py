from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from pypdf import PdfReader
import torch
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import json
import random
import os
from dotenv import load_dotenv
import os
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()


# Check if the token is set
token = os.getenv("HUGGING_FACE_TOKEN")

login(token)

# Load the Hugging Face token from the environment
token = os.getenv("HUGGING_FACE_TOKEN")
if not token:
    raise ValueError("HUGGING_FACE_TOKEN is not set in the environment")

class QuizGenerator:
    def __init__(self):  # Fixed double underscore
        model_name = "gpt2"  # Change to a publicly available model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def extract_pdf_content(self, pdf_file: UploadFile) -> str:
        """Extract text content from uploaded PDF."""
        reader = PdfReader(pdf_file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def generate_question(self, context: str, difficulty: float = 0.5) -> dict:
        """Generate a question based on the context and difficulty level."""
        prompt = f"""Based on this text: "{context}"

Generate a multiple-choice question with difficulty level {difficulty} (0-1).
Include:
1. The question
2. Four answer choices (A, B, C, D)
3. The correct answer
4. A brief explanation

Format the output as JSON with these keys:
- question
- options (array of 4 strings)
- correct_index (0-3)
- explanation

Make sure the incorrect options are plausible but clearly wrong to someone who understands the material.
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_length=800,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            json_str = response[response.find("{"):response.rfind("}") + 1]
            question_data = json.loads(json_str)

            return {
                "question": question_data["question"],
                "options": question_data["options"],
                "correct_index": question_data["correct_index"],
                "explanation": question_data["explanation"],
                "difficulty": difficulty
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return None


# Global variables
quiz_generator = None
user_profiles = {}


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global quiz_generator
    quiz_generator = QuizGenerator()
    yield
    # Shutdown: Clean up resources
    global user_profiles
    user_profiles.clear()


# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "Welcome to QuizGenerator API",
        "endpoints": {
            "docs": "/docs",
            "upload_pdf": "/upload-pdf/",
            "generate_question": "/generate-question/{user_id}",
            "submit_answer": "/submit-answer/"
        }
    }


class UserAnswer(BaseModel):
    user_id: str
    question_id: int
    selected_option: int
    time_taken: float


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), user_id: str = None):
    try:
        content = quiz_generator.extract_pdf_content(file)
        # Store content for later use in question generation
        if user_id:
            user_profiles[user_id] = {"content": content}
        return {"message": "PDF processed successfully", "content_length": len(content)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/generate-question/{user_id}")
async def generate_question(user_id: str):
    if user_id not in user_profiles or "content" not in user_profiles[user_id]:
        return {"error": "Please upload a PDF first"}
    return quiz_generator.generate_question(user_profiles[user_id]["content"])


@app.post("/submit-answer/")
async def submit_answer(answer: UserAnswer):
    return {"message": "Answer recorded successfully"}