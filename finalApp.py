from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from transformers import pipeline
# import fitz  # PyMuPDF
import PyPDF2


CHROMA_PATH = "chroma"

summarizer = pipeline("summarization")
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm_hf = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token= HUGGINGFACEHUB_API_TOKEN
)


Summariser_template = """

    Refine the provided context to be used as source of mcq question generation.
    {information}

"""

MCQ_Generator_template_text = """
Your task is to create 5 MCQ questions based on the context provided and the format should be Multiple choice with 4 options(a,b,c,d) and the correct answer. 

{contextt}

"""
llm = llm_hf



def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text



def summarize_text(text, max_length=150):
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def get_user_inputs():
    # Ask for the number of questions
    while True:
        try:
            num_questions = int(input("Enter the number of questions: "))
            if num_questions <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid positive integer for the number of questions.")
    
    # Ask for the difficulty level
    difficulty_levels = ['easy', 'medium', 'hard']
    while True:
        difficulty = input(f"Enter the difficulty level ({'/'.join(difficulty_levels)}): ").lower()
        if difficulty in difficulty_levels:
            break
        else:
            print(f"Please choose a valid difficulty level from {difficulty_levels}.")
    
    # Ask for specific context (optional)
    context = input("Enter specific context for the questions (press Enter to use the summary of the document): ")
    print(context)
    print(type(context))
    return num_questions, difficulty, context

def create_prompt(num_questions, difficulty, context):
    if context:
        prompt = (f"Create {num_questions} {difficulty} multiple choice questions based on the following context:\n\n"
                  f"{context}")
    else:
        prompt = (f"Create {num_questions} {difficulty} multiple choice questions based on the summary of the document.")
    
    return prompt


def generate_mcq(prompt):
    
    Summarization_template = PromptTemplate.from_template(Summariser_template)
    MCQ_template = PromptTemplate.from_template(MCQ_Generator_template_text)

    # Create the chains.
    summarization_chain = LLMChain(llm=llm, prompt=Summarization_template)
    MCQ_chain = LLMChain(llm=llm, prompt=MCQ_template)

    # Join them into a sequential chain.
    overall_chain = SimpleSequentialChain(
        chains=[summarization_chain, MCQ_chain], verbose=True
    )

    overall_chain.run(prompt)


def search_db(query_text):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # changed no of queries from 3 to 1
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=1)

    return results
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def main(pdf_path):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Get user inputs
    num_questions, difficulty, context = get_user_inputs()

    if context is not None:
        results = search_db(context)
        

    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        results = summarize_text(pdf_text)
        # return

    
    # # Generate context if not provided
    # if not context:
    #     context = summarize_text(pdf_text)
    
    # Create the final prompt
    prompt = create_prompt(num_questions, difficulty, results)
    


    # Generate MCQ quiz
    # mcq_quiz = generate_mcq(prompt, num_questions)
    
    mcq_quiz = generate_mcq(prompt)

    print(mcq_quiz)
    # # Print the generated questions
    # for idx, choice in enumerate(mcq_quiz, 1):
    #     print(f"Question {idx}:\n{choice.text}\n")

# Example usage
pdf_path = "Data/1111India.pdf"

main(pdf_path)

