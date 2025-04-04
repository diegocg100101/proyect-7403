import ollama

# Definir el modelo y el prompt
modelo = "deepseek-r1:8b" 

ruta = 'contexto.txt'

with open(ruta, 'r', encoding="utf-8") as file:
    contexto = file.read()

ollama.chat(model=modelo, messages=[{"role": "user", "content": contexto}])

while True:
    prompt = input(">")
 
    # Enviar el prompt al modelo
    respuesta = ollama.chat(model=modelo, messages=[{"role": "user", "content": prompt}])

    # Mostrar la respuesta
    print(respuesta['message']['content'])
