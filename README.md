# QA-LLM
Multiple-choice Exam Model
This project is focused on answering multiple-choice questions, particularly in the medical domain, using advanced NLP techniques. The model is designed to handle datasets like MedMCQA and MMLU, providing detailed, step-by-step reasoning to arrive at the correct answers.

# Model Architecture
Our model is built on llama3-8b, selected for its balance between high accuracy and run-time efficiency. This architecture allows the model to handle complex medical questions while maintaining efficient performance.

# Training Data
The model was trained using the following datasets:

Ultramedical Dataset: A comprehensive dataset covering various medical specialties.\
Dementia Dataset: A synthetic dataset generated using GPT-4, focusing on dementia-related cases. This dataset enhances the model's ability to understand and process neurological medical information.
Prompt Template
The model answers multiple-choice questions using the following template:

python
코드 복사
prompt = """Below is a question with multiple choice options. Provide a detailed answer based on the information provided. Use step by step reasoning to arrive at the correct answer.

## Question
{question}

## Options:
{options}

## Task
Answer the above question with format 'So, the answer is' after your explanation. For example, if the answer is A, write 'So, the answer is A'.

### Response:
"""
Example Usage
Here's an example code that demonstrates how to use the model for answering a multiple-choice question:

python
코드 복사
prompt = """Below is a question with multiple choice options. Provide a detailed answer based on the information provided. Use step by step reasoning to arrive at the correct answer.

## Question
{question}

## Options:
{options}

## Task
Answer the above question with format 'So, the answer is' after your explanation. For example, if the answer is A, write 'So, the answer is A'.

### Response:
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("your-model/llama3-8b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("your-model/llama3-8b")

question = "What is the most common cause of dementia?"
options = "A) Alzheimer's disease\nB) Parkinson's disease\nC) Vascular dementia\nD) Lewy body dementia"

model_input = prompt.format(question=question, options=options)
input_ids = tokenizer(model_input, return_tensors="pt").input_ids
output = model.generate(input_ids)
print(tokenizer.decode(output[0]))
"""
In this example, the model generates an answer based on a multiple-choice question about the most common cause of dementia. The output will include a detailed reasoning process followed by the final answer in the format: 'So, the answer is A.'

Features
Multiple-choice exam solving: The model provides detailed reasoning for each answer choice.
High performance: Leveraging llama3-8b, the model strikes a balance between accuracy and efficiency.
Specialized medical knowledge: The model is trained on datasets tailored for the medical domain, enhancing its expertise in this field.
Contributing
We welcome contributions from the community! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request once your changes are complete.
Ensure all tests pass before submitting.
License
This project is licensed under the MIT License. For more details, see the LICENSE file.
