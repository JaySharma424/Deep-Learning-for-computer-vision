import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfer_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="DeepFER — Facial Emotion Recognition",
    page_icon="🙂",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    return image, arr

def predict(image):
    resized, arr = preprocess_image(image)
    probs = model.predict(arr[None, ...], verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    return resized, probs, top_indices

def gradcam_overlay(image_array, conv_layer_name="top_conv"):
    backbone = model.get_layer("efficientnetb0")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            backbone.get_layer(conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array[None, ...])
        class_idx = tf.argmax(predictions[0])
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_rgb = np.asarray(
        tf.keras.utils.array_to_img(
            tf.keras.applications.imagenet_utils.preprocess_input(
                np.asarray(heatmap_img.convert("RGB"), dtype=np.float32)
            )
        )
    )

    # Simpler red-channel intensity overlay for robustness.
    hm = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    base = image_array.astype(np.float32) / 255.0
    overlay = np.empty_like(base)
    overlay[..., 0] = np.clip(base[..., 0] * 0.55 + hm * 0.45, 0, 1)
    overlay[..., 1] = base[..., 1] * 0.65
    overlay[..., 2] = base[..., 2] * 0.65

    return overlay

st.title("DeepFER")
st.caption("Facial-expression recognition using EfficientNetB0 transfer learning")

st.info(
    "This model predicts visual expression categories from an image. "
    "It is not a diagnostic tool and should not be used to infer a person's "
    "mental-health condition or internal emotional state."
)

tab_upload, tab_camera, tab_about = st.tabs(
    ["Upload Image", "Camera", "Project Details"]
)

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        resized, probs, top_indices = predict(image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Input image", use_container_width=True)

        with col2:
            best_idx = int(top_indices[0])
            st.subheader(f"Predicted expression: {class_names[best_idx].title()}")
            st.metric("Model confidence", f"{probs[best_idx] * 100:.2f}%")

            top_table = {
                "Expression": [class_names[i].title() for i in top_indices],
                "Confidence": [f"{probs[i] * 100:.2f}%" for i in top_indices]
            }
            st.table(top_table)

            st.bar_chart(
                {class_names[i].title(): float(probs[i]) for i in top_indices}
            )

        show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

        if show_gradcam:
            try:
                arr = np.asarray(resized, dtype=np.float32)
                overlay = gradcam_overlay(arr)
                st.image(
                    overlay,
                    caption="Grad-CAM activation overlay",
                    use_container_width=True
                )
            except Exception as exc:
                st.warning(f"Grad-CAM could not be generated for this model: {exc}")

with tab_camera:
    st.write("Capture a face image using your browser camera and run inference.")

    camera_file = st.camera_input("Take a photo")

    if camera_file:
        image = Image.open(camera_file)
        resized, probs, top_indices = predict(image)

        best_idx = int(top_indices[0])
        st.image(image, caption="Captured frame", use_container_width=True)
        st.success(
            f"Predicted expression: {class_names[best_idx].title()} "
            f"({probs[best_idx] * 100:.2f}%)"
        )

        st.write("Top 3 predictions")
        for idx in top_indices:
            st.write(
                f"**{class_names[idx].title()}** — "
                f"{probs[idx] * 100:.2f}%"
            )

with tab_about:
    st.subheader("DeepFER project")
    st.write(
        "DeepFER uses image preprocessing, augmentation, EfficientNetB0 "
        "transfer learning, class-weighted training, fine-tuning, and "
        "post-training error analysis."
    )

    st.subheader("Predicted classes")
    st.write(", ".join(name.title() for name in class_names))

    st.subheader("Real-world considerations")
    st.write(
        "Performance can change with lighting, camera quality, occlusion, "
        "pose, demographic composition, and domain shift. The model should "
        "be validated on representative deployment data before production use."
    )

st.divider()
st.caption("DeepFER • CNN / Transfer Learning • Streamlit")
