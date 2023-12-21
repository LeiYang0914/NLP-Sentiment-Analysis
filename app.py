import streamlit as st
from googletrans import Translator

def translate(text, target_language='en'):
    translator = Translator()
    try:
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        return f"Error in translation: {e}"

def main():
    st.title("Translation App")

    text = st.text_area("Enter text to translate:")

    if st.button("Translate"):
        if text:
            translated_text = translate(text)
            st.write("Translated Text:", translated_text)
        else:
            st.write("Please enter some text to translate.")

if __name__ == "__main__":
    main()
