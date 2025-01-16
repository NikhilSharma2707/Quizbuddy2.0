from train_model import OptimizedQuizGenerator


def main():
    # Initialize the generator
    quiz_gen = OptimizedQuizGenerator()

    # Your context
    context = "Note: Principles of Financial Accounting and Principles of Managerial Accounting adopters may have received an email announcing the retirement of CNOWV2 for accounting. We are pleased to let you know that Lyryx's widely used online homework for accounting is now fully integrated with OpenStax Financial and Managerial Accounting. The program includes spreadsheets, algorithm problems, and a high degree of customization. Please visit Lyryx to learn more.Principles of Accounting is designed t"


    # Generate a question
    question = quiz_gen.generate_question(
        context=context,
        difficulty="medium"
    )

    print("Generated Question:", question)


if __name__ == "__main__":
    main()