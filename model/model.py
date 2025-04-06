from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Cargar modelo y tokenizer entrenados
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

# Crear pipeline de generación
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Usuario: ¿Qué opinas de los humanos?\nIA:"
result = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
print(result[0]["generated_text"])