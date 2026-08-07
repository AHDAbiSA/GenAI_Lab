from transformers import pipeline

print("=" * 60)
print("TEXT SUMMARIZATION")
print("=" * 60)

# Load summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Input text
text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that normally require human intelligence.
It is widely used in healthcare, education, manufacturing, finance,
transportation, and cybersecurity. AI systems can analyze large
amounts of data, identify patterns, make predictions, and support
intelligent decision-making. Generative AI is a branch of Artificial
Intelligence that can create new content such as text, images, audio,
video, and computer programs.
"""

# Generate summary
summary = summarizer(
    text,
    max_length=60,
    min_length=20,
    do_sample=False
)

print("\nSummary:\n")
print(summary[0]["summary_text"])

print("\n" + "=" * 60)
print("QUESTION ANSWERING")
print("=" * 60)

# Load Question Answering model
question_answerer = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Context
context = """
Generative Artificial Intelligence is a type of Artificial Intelligence
that can create new content such as text, images, audio, video, and
computer programs. Large Language Models are commonly used for text
generation, summarization, translation, and question answering.
"""

# Question
question = "What type of content can Generative AI create?"

# Predict answer
result = question_answerer(
    question=question,
    context=context
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nConfidence Score:")
print(round(result["score"], 3))