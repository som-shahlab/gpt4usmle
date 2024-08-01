import openai
import json
import random

# Initialize the OpenAI client
client = openai.OpenAI(api_key=API_KEY)

# Function to run GPT-4 on a query
def run_gpt4_test(query):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a member of the USMLE test committee for USMLE Step 2."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to categorize a question with chain-of-thought reasoning
def categorize_question(question, answer, categories):
    prompt = (
        f"The following is a question from the USMLE Step 2 CK exam:\n\n"
        f"{question}\n\n"
        f"Correct Answer: {answer}\n\n"
        f"To correctly answer this question, first think about what information you would need to know to arrive at the correct answer. "
        f"Then provide the category based on the information a student would need to know to arrive at the correct answer. "
        f"Choose one of the following broad categories: {', '.join(categories)}.\n\n"
        f"Category:"
    )
    return run_gpt4_test(prompt)

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Load questions from JSON file
    file_path = '../data/matched_questions_and_answers.json'
    questions = load_json(file_path)
    
    # List of categories
    categories = [
        "Human Development", "Immune System", "Blood & Lymphoreticular System",
        "Behavioral Health", "Nervous System & Special Senses", "Skin & Subcutaneous Tissue",
        "Musculoskeletal System", "Cardiovascular System", "Respiratory System",
        "Gastrointestinal System", "Renal & Urinary System",
        "Pregnancy, Childbirth, & the Puerperium", 
        "Female and Transgender Reproductive System & Breast",
        "Male and Transgender Reproductive System", "Endocrine System",
        "Multisystem Processes & Disorders",
        "Biostatistics, Epidemiology/Population Health, & Interpretation of the Medical Literature",
        "Social Sciences"
    ]
    print("Categories provided:", categories)

    # Categorize each question
    categorized_questions = []
    for entry in questions:
        input_index = entry['input_index']
        output_index = entry['output_index']
        input_question = entry['input_question']
        input_answer = entry['input_answer']
        generated_question = entry['generated_question']
        generated_answer = entry['generated_answer']
        
        input_category = categorize_question(input_question, input_answer, categories)
        generated_category = categorize_question(generated_question, generated_answer, categories)
        
        categorized_questions.append({
            'input_index': input_index,
            'output_index': output_index,
            'input_question': input_question,
            'generated_question': generated_question,
            'input_category': input_category,
            'generated_category': generated_category
        })

    # Save the categorized questions to a new JSON file
    output_file_path = 'categorized_questions_with_answers_usmle.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(categorized_questions, f, indent=4, ensure_ascii=False)

    print(f"Questions have been categorized and saved to '{output_file_path}'.")
