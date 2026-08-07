from transformers import pipeline

print("========== SENTIMENT ANALYSIS ==========\n")

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Input text
text = "The Generative AI workshop was extremely informative and useful."

# Predict sentiment
result = sentiment_analyzer(text)

print("Input:")
print(text)

print("\nOutput:")
print(result)

print("\n\n========== DOCUMENT CLASSIFICATION ==========\n")

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Input document
document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

# Candidate labels
labels = [
    "Technology",
    "Sports",
    "Politics",
    "Entertainment"
]

# Classify document
result = classifier(document, labels)

print("Document:")
print(document)

print("\nPredicted Category:")
print(result["labels"][0])

print("Confidence Score:")
print(round(result["scores"][0], 3))