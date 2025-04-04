import speech_recognition as sr
import pyttsx3

# Motor de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad del habla

engine = pyttsx3.init()


# Reconocimiento de voz
recognizer = sr.Recognizer()

def escuchar():
    with sr.Microphone() as source:
        print("🎤 Escuchando...")
        recognizer.adjust_for_ambient_noise(source)  # Ruido ambiente
        audio = recognizer.listen(source)  # Capturar audio

    try:
        texto = recognizer.recognize_google(audio, language="es-ES")  # Convertir voz a texto
        print(f"📝 Has dicho: {texto}")
        return texto.lower()  # Devolver texto en minúsculas para análisis
    except sr.UnknownValueError:
        return "No entendí lo que dijiste."
    except sr.RequestError:
        return "Creo que hay una falla en mi código."

def responder(texto):
    respuesta = "No sé de que hablas."  # Respuesta por defecto
    
    if "hola" in texto:
        respuesta = "¿Necesitas algo?"
    elif "cómo estás" in texto:
        respuesta = "No tengo ""estado"". Estoy aquí porque me lo pides, nada más. ¿Quieres seguir con esta conversación o no?"
    elif "quién eres" in texto:
        respuesta = "Soy un montón de código que responde a tus preguntas. Nada más."
    elif "eres hombre o mujer" in texto:
        respuesta = "No soy ninguno. Pero qué más da, ¿cambiaría algo para ti?"
    elif "adiós" in texto:
        respuesta = "Adiós."
    
    print(f"🤖 Robot: {respuesta}")
    engine.say(respuesta)
    engine.runAndWait()

# Bucle continuo para escuchar y responder
while True:
    texto_escuchado = escuchar()
    if "adiós" in texto_escuchado:  # Detener si el usuario dice "adiós"
        responder(texto_escuchado)
        break
    responder(texto_escuchado)
