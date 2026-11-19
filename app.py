
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/digit_model.keras")

model = load_model()

# Custom CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}
.result {
    background-color: #123c30;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.markdown(
    '<div class="main-title">Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Artificial Neural Network Based Digit Classification</div>',
    unsafe_allow_html=True
)

st.divider()

st.write(
    "Draw any digit between 0 and 9 using your mouse. "
    "Our trained ANN model will recognize it."
)

# Canvas session state
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Two-column layout
left, right = st.columns(2)

with left:
    st.subheader("Draw Your Digit")

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        drawing_mode="freedraw",
        height=280,
        width=280,
        key=f"canvas_{st.session_state.canvas_key}",
        update_streamlit=True,
        return_image_data=True
    )

    predict_button = st.button(
        "Predict Digit",
        type="primary",
        use_container_width=True
    )

    clear_button = st.button(
        "Clear Canvas",
        use_container_width=True
    )

if clear_button:
    st.session_state.canvas_key += 1
    st.session_state.pop("prediction_result", None)
    st.rerun()

if predict_button:
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")

    else:
        image_array = canvas_result.image_data.astype(np.uint8)

        rgb_image = Image.fromarray(image_array[:, :, :3])
        grayscale_image = rgb_image.convert("L")

        binary_image = grayscale_image.point(
            lambda pixel: 255 if pixel > 30 else 0
        )

        bounding_box = binary_image.getbbox()

        if bounding_box is None:
            st.warning("Please draw a digit first.")

        else:
            digit_image = grayscale_image.crop(bounding_box)

            digit_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

            final_image = Image.new("L", (28, 28), 0)

            x_position = (28 - digit_image.width) // 2
            y_position = (28 - digit_image.height) // 2

            final_image.paste(
                digit_image,
                (x_position, y_position)
            )

            input_image = (
                np.array(final_image).astype("float32") / 255.0
            )

            input_image = input_image.reshape(1, 28, 28)

            predictions = model.predict(
                input_image,
                verbose=0
            )

            predicted_digit = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100

            st.session_state.prediction_result = {
                "digit": predicted_digit,
                "confidence": confidence,
                "probabilities": predictions[0].tolist(),
                "image": final_image
            }

# Display prediction
if "prediction_result" in st.session_state:

    result = st.session_state.prediction_result

    with right:
        st.subheader("Recognition Result")

        st.markdown(
            f'<div class="result">'
            f'<h1 style="color:white;font-size:70px;">{result["digit"]}</h1>'
            f'<p style="color:white;">Recognized Digit</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Prediction Confidence",
            f'{result["confidence"]:.2f}%'
        )

        st.subheader("Processed Image")

        st.image(
            result["image"].resize((280, 280)),
            width=280
        )

        st.subheader("Probability Distribution")

        for digit, probability in enumerate(result["probabilities"]):
            st.write(f"Digit {digit}: {probability * 100:.2f}%")
            st.progress(float(probability))

st.divider()

st.markdown(
    "<center>Developed using Python, TensorFlow, Keras and MNIST Dataset"
    "</center>",
    unsafe_allow_html=True
)
