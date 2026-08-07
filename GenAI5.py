from transformers import pipeline

# Load pretrained instruction-following model
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

# Function to generate response
def generate_response(prompt, max_tokens=200):
    result = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    return result[0]["generated_text"]


print("=" * 60)
print("PROMPT ENGINEERING APPLICATION")
print("=" * 60)

# --------------------------------------------------
# 1. Content Generation
# --------------------------------------------------

content_prompt = """
Role: You are an experienced Artificial Intelligence teacher.

Task:
Write a simple introduction to Generative Artificial Intelligence.

Target audience:
First-year engineering students.

Requirements:
1. Use simple language.
2. Limit the response to five sentences.
3. Include two real-world applications.
4. Avoid highly technical terms.
"""

content_output = generate_response(content_prompt)

print("\n1. CONTENT GENERATION")
print("-" * 60)
print(content_output)

# --------------------------------------------------
# 2. Reasoning Task
# --------------------------------------------------

reasoning_prompt = """
Solve the following problem.

A college conducted a Generative AI workshop for 120 students.
Eighty-five students completed the workshop successfully.

Instructions:
1. Identify the total number of students.
2. Identify the completed students.
3. Calculate students who did not complete.
4. Give the final answer.
"""

reasoning_output = generate_response(reasoning_prompt)

print("\n2. REASONING TASK")
print("-" * 60)
print(reasoning_output)

# --------------------------------------------------
# 3. Email Generation
# --------------------------------------------------

email_prompt = """
Role: You are a professional academic coordinator.

Task:
Write a formal email to students.

Context:
A Generative AI laboratory session is scheduled for Friday at
10:00 AM in AI Laboratory 2.

Students must bring laptops and complete Hugging Face registration.

Requirements:
1. Include subject.
2. Professional tone.
3. Mention venue and time.
4. Keep it concise.
"""

email_output = generate_response(email_prompt)

print("\n3. EMAIL GENERATION")
print("-" * 60)
print(email_output)

# --------------------------------------------------
# 4. Action Item Extraction
# --------------------------------------------------

meeting_notes = """
The project team reviewed the development of the college chatbot.

Arun will prepare the training dataset by Monday.

Priya will test the chatbot responses by Wednesday.

Rahul will prepare the final demonstration and presentation.

The team will meet again on Friday.
"""

action_prompt = f"""
Extract action items from the meeting notes.

Meeting Notes:

{meeting_notes}

For each action item provide:

1. Person Responsible
2. Assigned Task
3. Deadline

Present as a numbered list.
"""

action_output = generate_response(action_prompt)

print("\n4. ACTION ITEM EXTRACTION")
print("-" * 60)
print(action_output)