from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load pretrained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt
prompt = "Artificial Intelligence is"

# Convert prompt into tokens
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

# Decode generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(generated_text)