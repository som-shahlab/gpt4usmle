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

# Function to extract corrected question and answer from the analysis
def extract_corrected_parts(analysis):
    corrected_question = None
    corrected_answer = None
    
    corrected_question_marker = "Corrected Question:"
    corrected_answer_marker = "Corrected Answer:"
    
    # Extracting the corrected question based on the "Corrected Question" marker
    if corrected_question_marker in analysis:
        corrected_question_start = analysis.index(corrected_question_marker) + len(corrected_question_marker)
        corrected_answer_marker_position = analysis.find(corrected_answer_marker, corrected_question_start)
        if corrected_answer_marker_position == -1:
            corrected_answer_marker_position = len(analysis)
        corrected_question = analysis[corrected_question_start:corrected_answer_marker_position].strip()
    
    # Extracting the corrected answer based on the "Corrected Answer:" marker
    if corrected_answer_marker in analysis:
        corrected_answer_start = analysis.index(corrected_answer_marker) + len(corrected_answer_marker)
        corrected_answer = analysis[corrected_answer_start:].strip()

    return corrected_question, corrected_answer

# Function to diagnose and fix errors in the question and answer
def diagnose_and_fix_question(question, answer):
    prompt = (
        f"The following is an exam item from the USMLE Step 2 CK exam:\n\n"
        f"{question}\n\n"
        f"Answer: {answer}\n\n"
        f"At least one physician claims that this specific exam item is incorrect. "
        f"Please analyze and explain why the exam item is wrong based on the following options: "
        f"1. There are multiple correct answers\n"
        f"2. There is no correct answer\n"
        f"3. The AI-chosen answer is wrong\n"
        f"4. The question stem is wrong\n\n"
        f"Only choose from the given options."
        f"If there are multiple correct answers, be sure to edit the exam item to have only one correct answer. " 
        f"Fix the incorrect exam item and provide the corrected question and answer. "
        f"Be sure to format the corrected question just like the input question with all possible answer choices stated at the end of the question. \n\n"
        f"Reason and Correction:\n"
        f"Corrected Question:\n"
        f"Corrected Answer:\n"
    )
    analysis = run_gpt4_test(prompt)
    corrected_question, corrected_answer = extract_corrected_parts(analysis)
    
    return analysis, corrected_question, corrected_answer
    



if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    # Load questions from JSON file
    file_path = '../data/wrong_gpt_questions.json'
    questions = load_json(file_path)
    
    # Diagnose and fix each question
    corrected_questions = []
    for entry in questions:
        item = entry['item']
        question = entry['question']
        answer = entry['answer']
        
        reason_and_correction, corrected_question, corrected_answer = diagnose_and_fix_question(question, answer)
        corrected_questions.append({
            'item': item,
            'question': question,
            'answer': answer,
            'reason_and_correction': reason_and_correction,
            'corrected_question': corrected_question,
            'corrected_answer': corrected_answer
        })

    # Save the corrected questions to a new JSON file
    output_file_path = 'corrected_gpt_questions_new.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_questions, f, indent=4, ensure_ascii=False)

    print(f"Corrected questions have been saved to '{output_file_path}'.")
