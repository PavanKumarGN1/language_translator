import asyncio
import concurrent.futures
import time
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import speech_recognition as sr
from gtts import gTTS
import os
import platform
import webbrowser

# Speech to text
def convert_speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.write("You said:", text)
        return text  # Return the recognized text
    except sr.UnknownValueError:
        st.write("Sorry, could not understand audio.")
    except sr.RequestError as e:
        st.write(f"Error connecting to Google API: {e}")
    return None

# Load the model and tokenizer
def initialize_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = initialize_model()

# Function to perform translation for a single chunk
def translate_chunk(chunk):
    tokenized_chunk = tokenizer(chunk, return_tensors="pt")
    generated_tokens = model.generate(**tokenized_chunk, forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])
    translated_text_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text_chunk

# Function to perform translation for longer sentences using parallelization
def translate_long_text_parallel(long_text, chunk_size=512):
    # Set source language to English
    tokenizer.src_lang = "en_XX"

    # Split the long English text into smaller chunks
    chunks = [long_text[i:i+chunk_size] for i in range(0, len(long_text), chunk_size)]

    # Use concurrent.futures to parallelize translation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Measure time for translation
        start_time = time.time()

        # Translate chunks concurrently
        translated_chunks = list(executor.map(translate_chunk, chunks))

        end_time = time.time()
        #st.write(f"Total time taken for translation: {end_time - start_time:.4f} seconds")

    # Concatenate the translated chunks to get the complete translation
    translated_text = " ".join(translated_chunks)

    return translated_text

# Text to speech part
def text_to_speech(hindi_text, lang="hi"):
    tts = gTTS(text=hindi_text, lang=lang, slow=False)
    tts.save("output.mp3")

    if os.path.exists("output.mp3"):
        st.write("Audio file saved successfully.")
        st.audio("output.mp3", format="audio/mp3")
    else:
        st.write("Error: Audio file not found.")

# Main code
if __name__ == "__main__":
    # Streamlit UI
    st.title("Voice To Voice Translation App")
    st.write("Click the button and speak something in English to get audio output in Hindi.")

    # Button to execute the code
    if st.button("Speak and Translate"):
        # Speech to text input
        text_input = convert_speech_to_text()

        if text_input:
            # Perform parallelized translation for longer sentences with adjusted chunk size
            translated_text = translate_long_text_parallel(text_input, chunk_size=512)

            # Print the translated text
            st.write("Translated text is:", translated_text)

            # Generate the audio file
            text_to_speech(translated_text)
        else:
            st.write("Speech recognition did not return any text.")
