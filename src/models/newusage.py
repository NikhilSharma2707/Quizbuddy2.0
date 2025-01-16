from Quiz_Generator import QuizGenerator
quiz_gen = QuizGenerator()

context = """
The solar system consists of the Sun and all celestial objects bound to it by gravity, 
including planets, dwarf planets, and numerous smaller bodies like asteroids and comets. 
Mercury is the closest planet to the Sun, while Neptune is currently the farthest planet.
"""

questions = quiz_gen.generate_question(
    context=context,
    difficulty="medium",
    question_type="mcq",
    num_questions=1
)