!pip install accelerate

import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the environment has access to a CUDA-capable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer directly to GPU if available
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True, device_map="auto")

# Define templates for problems
templates = {
    "algebra": {
        "easy": ["Solve for x: {a}x + {b} = {c}", "Find the value of x: {a}x - {b} = {c}"],
        "medium": ["Solve for x: {a}x^2 + {b}x + {c} = 0", "Find the roots of: {a}x^2 - {b}x = {c}"],
        "hard": ["Solve for x: {a}x^3 + {b}x^2 + {c}x + {d} = 0", "Find the value of x in the equation: {a}x^3 - {b}x^2 + {c}x = {d}"]
    },
    "calculus": {
        "easy": ["Differentiate the function: f(x) = {a}x^2 + {b}x + {c}", "Find the derivative of: f(x) = {a}x^3 - {b}x + {c}"],
        "medium": ["Integrate the function: f(x) = {a}x^2 + {b}x + {c}", "Find the integral of: f(x) = {a}x^3 - {b}x + {c}"],
        "hard": ["Solve the differential equation: {a}dy/dx + {b}y = {c}", "Find the solution to the differential equation: {a}d^2y/dx^2 - {b}dy/dx + {c}y = 0"]
    }
    # Add more areas and difficulties as needed
}

def generate_synthetic_math_problems(num_problems):
    problems = []

    for _ in range(num_problems):
        # Randomly choose an area of mathematics
        area = random.choice(list(templates.keys()))
        
        # Randomly choose a difficulty level
        difficulty = random.choice(list(templates[area].keys()))
        
        # Randomly choose a template
        template = random.choice(templates[area][difficulty])
        
        # Randomly generate parameters
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        d = random.randint(1, 10)
        
        # Generate the problem using the template and parameters
        problem = template.format(a=a, b=b, c=c, d=d)
        problems.append(problem)
    
    return problems

def solve_problem(problem):
    # Encode the problem
    inputs = tokenizer(problem, return_tensors="pt").to(device)
    
    # Generate a response from the model
    outputs = model.generate(inputs["input_ids"], max_length=100)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip the answer to only the math (assuming answer is preceded by "The answer is ")
    if "The answer is " in response:
        answer = response.split("The answer is ")[-1].strip()
    else:
        answer = response.strip()
    
    return answer

def generate_and_solve_problems(num_problems):
    problems = generate_synthetic_math_problems(num_problems)
    solved_problems = []

    for problem in problems:
        answer = solve_problem(problem)
        solved_problems.append({
            "problem": problem,
            "answer": answer
        })

    return solved_problems

def main(num_problems):
    solved_problems = generate_and_solve_problems(num_problems)
    return json.dumps(solved_problems, indent=4)

# Example usage
num_problems = 10
output = main(num_problems)
print(output)
