import os
from flask import Flask, render_template, request, send_file, jsonify
import pdfplumber
import docx
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from fpdf import FPDF
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Initialize device and model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    # Use a smaller model instead of Llama-2-7b
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much smaller model
    # Alternative small models you could try:
    # MODEL_NAME = "facebook/opt-125m"
    # MODEL_NAME = "EleutherAI/pythia-160m"

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load model with memory-efficient settings
    logger.info("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    # Additional memory optimizations
    model.config.use_cache = False

    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error("\nPossible solutions:")
    logger.error("1. Check your Hugging Face token and permissions")
    logger.error("2. Ensure you have enough system memory")
    logger.error("3. Try using an even smaller model")
    raise

# Store user performance data
user_performance = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_file(file_path):
    try:
        ext = file_path.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ''.join([page.extract_text() or '' for page in pdf.pages])
            return text
        elif ext == 'docx':
            doc = docx.Document(file_path)
            text = ' '.join([para.text for para in doc.paragraphs])
            return text
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return None
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        return None


def generate_quiz_questions(content, num_questions, difficulty='medium', previous_performance=None):
    try:
        # Memory optimization before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Adapt prompt based on user's performance
        difficulty_adjustment = ""
        if previous_performance:
            if previous_performance['accuracy'] < 0.6:
                difficulty_adjustment = "Make questions slightly easier. Focus on fundamental concepts."
            elif previous_performance['accuracy'] > 0.8:
                difficulty_adjustment = "Make questions more challenging. Include some advanced concepts."

        prompt = f"""
        Generate {num_questions} multiple-choice questions based on this content:
        '{content}'

        Difficulty level: {difficulty}
        {difficulty_adjustment}

        For each question:
        1. Create a clear question
        2. Generate 4 options (A, B, C, D)
        3. Ensure one correct answer
        4. Create plausible distractors
        5. Mark the correct answer

        Format each question as JSON:
        {{
            "question": "question text",
            "options": {{
                "A": "option A",
                "B": "option B",
                "C": "option C",
                "D": "option D"
            }},
            "correct_answer": "A",
            "explanation": "brief explanation"
        }}

        Return all questions in a JSON array.
        """

        # Generate response using Llama with proper error handling
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Parse and validate JSON response
            questions = json.loads(response)
            if not isinstance(questions, list):
                questions = [questions]
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing model output as JSON: {str(e)}")
            return []

    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return []


def update_user_performance(user_id, question_results):
    try:
        if user_id not in user_performance:
            user_performance[user_id] = {
                'total_questions': 0,
                'correct_answers': 0,
                'topic_performance': {},
                'difficulty_level': 'medium',
                'accuracy': 0
            }

        performance = user_performance[user_id]
        performance['total_questions'] += len(question_results)
        performance['correct_answers'] += sum(1 for q in question_results if q['correct'])

        # Update topic performance
        for result in question_results:
            topic = result.get('topic', 'general')
            if topic not in performance['topic_performance']:
                performance['topic_performance'][topic] = {
                    'correct': 0,
                    'total': 0
                }
            performance['topic_performance'][topic]['total'] += 1
            if result['correct']:
                performance['topic_performance'][topic]['correct'] += 1

        # Calculate and store accuracy
        performance['accuracy'] = performance['correct_answers'] / performance['total_questions']

        # Adjust difficulty based on recent performance
        if performance['accuracy'] > 0.8:
            performance['difficulty_level'] = 'hard'
        elif performance['accuracy'] < 0.6:
            performance['difficulty_level'] = 'easy'
        else:
            performance['difficulty_level'] = 'medium'

    except Exception as e:
        logger.error(f"Error updating user performance: {str(e)}")


def create_performance_report(user_id):
    try:
        if user_id not in user_performance:
            return None

        performance = user_performance[user_id]
        accuracy = (performance['correct_answers'] / performance['total_questions']) * 100

        report = FPDF()
        report.add_page()
        report.set_font("Arial", size=12)

        # Add report header
        report.cell(0, 10, "Performance Report", ln=True, align='C')
        report.ln(10)

        # Add overall statistics
        report.cell(0, 10, f"Overall Accuracy: {accuracy:.2f}%", ln=True)
        report.cell(0, 10, f"Total Questions Attempted: {performance['total_questions']}", ln=True)
        report.cell(0, 10, f"Current Difficulty Level: {performance['difficulty_level']}", ln=True)

        # Add topic-wise performance
        report.ln(10)
        report.cell(0, 10, "Topic-wise Performance:", ln=True)
        for topic, stats in performance['topic_performance'].items():
            topic_accuracy = (stats['correct'] / stats['total']) * 100
            report.cell(0, 10, f"{topic}: {topic_accuracy:.2f}%", ln=True)

        # Save report
        report_path = os.path.join(app.config['RESULTS_FOLDER'], f'report_{user_id}.pdf')
        report.output(report_path)
        return report_path

    except Exception as e:
        logger.error(f"Error creating performance report: {str(e)}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_quiz():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        content = extract_text_from_file(file_path)
        if not content:
            return jsonify({'error': 'Could not extract content from file'}), 400

        user_id = request.form.get('user_id', 'default_user')
        num_questions = int(request.form.get('num_questions', 5))

        previous_performance = user_performance.get(user_id)
        questions = generate_quiz_questions(
            content,
            num_questions,
            difficulty=previous_performance['difficulty_level'] if previous_performance else 'medium',
            previous_performance=previous_performance
        )

        # Clean up uploaded file
        os.remove(file_path)

        return jsonify({'questions': questions})

    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/submit', methods=['POST'])
def submit_answers():
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')
        question_results = data.get('results', [])

        update_user_performance(user_id, question_results)
        report_path = create_performance_report(user_id)

        if not report_path:
            return jsonify({'error': 'Could not generate report'}), 500

        return jsonify({
            'success': True,
            'report_url': f'/download/{os.path.basename(report_path)}'
        })

    except Exception as e:
        logger.error(f"Error submitting answers: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': 'File not found'}), 404


if __name__ == "__main__":
    app.run(debug=True)