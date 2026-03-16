import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, "../../.env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_aptitude_questions(jd_text: str, difficulty_level: str = "Medium", custom_instructions: str = "", mcq_count: int = 25):
    """
    Analyzes the Job Description and generates the specified number of MCQ questions and 4 Coding questions.
    Incorporates difficulty level and additional recruiter instructions.
    """
    
    prompt = f"""
    Create a recruitment assessment JSON based on the provided Job Description and specific difficulty requirements.
    
    DIFFICULTY LEVEL: {difficulty_level}
    TOTAL MCQs REQUIRED: {mcq_count}
    ADDITIONAL RECRUITER INSTRUCTIONS: {custom_instructions if custom_instructions else "None"}

    DIFFICULTY DEFINITIONS:
    - Low: Basic syntax, core concepts, entry-level definitions.
    - Medium: Application-based questions, 2-4 years industry experience level, common edge cases.
    - Hard: Advanced internals, complex logical reasoning, system design patterns, high-level algorithms, 5+ years expert level.

    REQUIRED JSON STRUCTURE:
    {{
      "mcqs": [
        {{
          "id": "Q1",
          "question": "text",
          "options": ["A", "B", "C", "D"],
          "answer": "correct option text"
        }}
      ],
      "coding_questions": [
        {{
          "title": "Title of DSA Problem",
          "description": "Clear problem statement and requirements",
          "constraints": "Complexity and input limits",
          "example_input": "sample input string",
          "example_output": "sample output string",
          "test_cases": [
            {{"input": "in1", "output": "out1"}},
            {{"input": "in2", "output": "out2"}}
          ]
        }}
      ]
    }}

    RULES:
    1. Generate exactly {mcq_count} MCQs.
    2. STATED DIFFICULTY IS MANDATORY. If "Hard" is selected, questions must be highly challenging and expert-level. Avoid trivial or basic questions.
    3. If the JD is technical (CS/IT), generate 4 Coding Questions. Otherwise, "coding_questions" must be [].
    4. Coding questions must match the {difficulty_level} level (e.g., Hard = Graph/Dynamic Programming, Low = Array/String manipulation).
    5. OUTPUT ONLY THE JSON. NO EXPLANATION.

    JOB DESCRIPTION:
    {jd_text}
    """

    print(f"\n--- 🚀 AGENT START: Analysing Job Description ---")
    try:
        print(f"Step 1: Connecting to OpenAI (GPT-4o)...")
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert technical interviewer and JSON generator. You strictly adhere to the requested DIFFICULTY LEVEL. For 'Hard', you provide complex, expert-level questions. For 'Medium', intermediate professional level. For 'Low', foundational basic level."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4000,
            response_format={ "type": "json_object" }
        )
        
        print(f"Step 2: Receiving AI response and parsing JSON...")
        response_content = completion.choices[0].message.content
        data = json.loads(response_content)
        
        # Log keys for verification
        if data.get("coding_questions") and len(data["coding_questions"]) > 0:
            print(f"DEBUG: Coding Question Keys: {list(data['coding_questions'][0].keys())}")
        
        mcqs = data.get("mcqs", [])
        coding = data.get("coding_questions", [])
        
        print(f"✅ SUCCESS: Generated {len(mcqs)} professional MCQs and {len(coding)} Coding questions.")
        return {"mcqs": mcqs, "coding_questions": coding}

    except Exception as e:
        print(f"❌ AGENT ERROR: {e}")
        raise e

def evaluate_code(problem_text: str, user_code: str, language: str, test_cases: list):
    """
    Evaluates the user's code against the problem statement and test cases using AI.
    """
    prompt = f"""
    Evaluate the following coding assessment submission.
    
    PROBLEM DESCRIPTION:
    {problem_text}
    
    TEST CASES:
    {json.dumps(test_cases, indent=2)}
    
    CANDIDATE CODE ({language}):
    {user_code}
    
    INSTRUCTIONS:
    1. Analyze the code for logic, correctness, and adherence to constraints.
    2. Check if the code would pass the provided test cases.
    3. Return a JSON object with:
       - "success": boolean (true if logic is correct and passes test cases)
       - "output": string (compiler-style output or explanation of errors)
       - "passed_count": number of test cases passed
       - "total_count": total number of test cases provided
    
    OUTPUT ONLY THE JSON.
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a code execution and evaluation agent. Be strict and accurate."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ EVALUATION ERROR: {e}")
        return {"success": False, "output": f"Evaluation Error: {str(e)}", "passed_count": 0, "total_count": len(test_cases)}
