from gptneo import HuggingFaceQuizGenerator

def demonstrate_quiz_generator():
    print("=== Hugging Face Quiz Generator Usage Demo ===")

    # Initialize the quiz generator
    quiz_gen = HuggingFaceQuizGenerator(model_name="gpt2")

    # Example contexts
    contexts = [
        """
        Photosynthesis is the process by which plants use sunlight, water and carbon dioxide
        to produce oxygen and energy in the form of sugar. It is one of the most important
        biochemical processes on Earth, providing the foundation for most life.
        """,
        """
        The theory of relativity, developed by Albert Einstein, describes how time, space,
        and gravity are interconnected. It consists of two parts: special relativity and
        general relativity. This theory revolutionized our understanding of the universe.
        """
    ]

    # Generate questions for each context with different difficulty levels
    for i, context in enumerate(contexts, 1):
        print(f"\n=== Context {i} ===")
        print(f"Context: {context.strip()}\n")

        for difficulty in ['easy', 'medium', 'hard']:
            result = quiz_gen.generate_quiz(context, difficulty)
            if result:
                print(f"{difficulty.capitalize()} Question: {result['question']}")
            else:
                print(f"Failed to generate {difficulty} question for this context.")

        print("\n" + "="*50)

    # Demonstrate how to generate a single question
    custom_context = """
    The Industrial Revolution was a period of significant economic, technological, and social changes that began in the late 18th century. It started in Britain and spread to other parts of the world. Major innovations like the steam engine, mechanized textile production, and the development of railroads marked this period.
    """
    print("\n=== Custom Question Generation ===")
    print(f"Context: {custom_context.strip()}\n")

    custom_question = quiz_gen.generate_question(custom_context, difficulty="medium")
    print(f"Generated Question: {custom_question}")

if __name__ == "__main__":
    demonstrate_quiz_generator()