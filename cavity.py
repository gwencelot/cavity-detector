import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import tempfile

# Define the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="IuJLxYcodlsBncCFCY7K"
)

MODEL_ID = "cavity-rs0uf/1"

def infer_image(image):
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
    
    # Use the inference client to send the image
    result = CLIENT.infer(temp_file_path, model_id=MODEL_ID)
    return result

# Set page title and favicon
st.set_page_config(page_title="Cavity Detection App", page_icon="🦷")

# Define app title and subtitle with emojis
st.title("🦷 Cavitri")
st.write("<span style='color:white;'>Upload an Image of your teeth to get started!</span>", unsafe_allow_html=True)


# Custom CSS to change background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #3F8E3F;  /* Change this value to the desired background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add space for better layout
st.write("")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Displaying preloader while detecting cavities
    with st.spinner("Detecting cavities..."):
        result = infer_image(image)

    # Display results after preloader
    if result and "predictions" in result:
        # Draw bounding boxes on the annotated image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Customize label size percentage
        label_size_percentage = 0.1  # Adjust this value to change the label size

        for prediction in result["predictions"]:
            if all(key in prediction for key in ["x", "y", "width", "height", "class", "confidence"]):
                x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
                class_label = prediction["class"]
                confidence = prediction["confidence"]
                # Adjust coordinates to center the bounding box
                x1, y1 = x - width / 2, y - height / 2
                x2, y2 = x + width / 2, y + height / 2
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Calculate text size and position
                font_size = int(height * label_size_percentage)  # Set font size to label_size_percentage of the bounding box height
                try:
                    font = ImageFont.truetype("arial.ttf", size=font_size)
                except OSError:
                    font = ImageFont.load_default()
                text = f"CAVITY: {class_label.upper()} ({confidence:.2f})"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
                text_x1, text_y1 = x1, y1 - text_size[1] - 5  # Position above the bounding box
                text_x2, text_y2 = text_x1 + text_size[0] + 10, text_y1 + text_size[1] + 5
                # Draw background rectangle for text
                draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill="green")  # Changed background color to green
                # Draw text on top of the background rectangle
                draw.text((text_x1 + 5, text_y1 + 2), text, fill="white", font=font)
                
        # Display annotated image
        st.subheader("🖼️ Annotated Image with Bounding Boxes")
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
        
        # Display number of cavities detected
        num_cavities = len(result["predictions"])
        st.subheader(f"Number of Cavities Detected: {num_cavities}")
        
        # Display bounding box results with smaller text and background color
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            '<h2 style="color: black;">🔍 Cavity Box Results</h2>',
            unsafe_allow_html=True
        )
        for idx, prediction in enumerate(result["predictions"]):
            cavity_class = prediction["class"]
            confidence = prediction["confidence"]
            st.markdown(
                f'<div style="background-color: green; padding: 10px; border-radius: 5px; font-size: 16px; color: white;">'  # Changed background color to green
                f'CAVITY: {cavity_class.upper()} (Confidence: {confidence:.2f})</div>',
                unsafe_allow_html=True
            )
        
    else:
        st.markdown('<p style="color: green; font-weight: bold;">No cavities detected. 😔</p>', unsafe_allow_html=True)
