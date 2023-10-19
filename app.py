from transformers import MusicgenForConditionalGeneration
from transformers import AutoProcessor
import scipy

import streamlit as st

# initialise model
@st.cache
def initialise_model():
    try:
        #processor = AutoProcessor.from_pretrained("musicgen-small")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        # model = MusicgenForConditionalGeneration.from_pretrained("musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        return processor, model
    except Exception as e:
        st.error(f"Error initializing the model: {str(e)}")
        return None, None
    

# Generate audio with given prompt
def generate_audio(processor, model, prompt):
    if processor is not None and model is not None:
        try:
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            audio_values = model.generate(**inputs.to("cpu"), do_sample=True, guidance_scale=3, max_new_tokens=256)
            return audio_values
        except Exception as e:
            st.error(f"Error generating audio: {str(e)}")
    return None

# save audio file with scipy
def save_file(model, audio_values, filename):
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

st.set_page_config(
    page_title="Plant Orchestra with GenAI",
    page_icon="🎵"
)

st.title("Plant Orchestra 🌻")
st.markdown("Generate music based on your own plant orchestra.")


prompt = st.text_input(label='Prompt:', value='Sunflower temperature: 32.5C UV light intensity: 50% Soil water level: 3cm/h')
if st.button("Generate Music"):
    with st.spinner("Initialising model..."):
        processor, model = initialise_model()
    if processor is not None and model is not None:
        with st.spinner("Generating audio..."):
            results = generate_audio(processor, model, prompt)
        if results is not None:
            with st.spinner("Saving audio..."):
                filename = "plant_orchestra" + ".wav"
                save_file(model, results, filename)
            with st.spinner("Displaying audio..."):
                with open(filename, "rb") as f:
                    generation = f.read()
                    st.write("Listen to the generated music:")
                st.audio(generation)
            
# Add additional information and instructions for users
st.sidebar.subheader("How to Use:")
st.sidebar.write("1. Enter a plant condition prompt in the text input.")
st.sidebar.write("2. Click the 'Generate Music' button to create music based on the provided prompt.")
st.sidebar.write("3. You can listen to the generated music and download it.")

# Footer
st.write()
st.write()
st.write()
st.markdown("---")
st.markdown("Created with ❤️ by HS2912 W4 Group 2")
