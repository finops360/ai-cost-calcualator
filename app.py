import yaml
import tiktoken
import streamlit as st
import pandas as pd

from utils import calculate_vision_token_cost

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="OpenAI API Pricing Calculator",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

# Load model attributes from YAML file
with open("config.yaml", "r") as file:
    models_data = yaml.safe_load(file)

# Sidebar for selecting model type
model_type = st.sidebar.selectbox("Select Type", ["Language Models"])

# Get the selected models based on the type
model_classes = models_data.get(model_type, [])

# Sidebar for selecting specific model class
selected_model_class = st.sidebar.selectbox(
    "Select Model Class", [model_class["name"] for model_class in model_classes]
)
select_model_class_idx = next(
    i
    for i, model_class in enumerate(model_classes)
    if model_class["name"] == selected_model_class
)

# Get the models for the selected model class
models = model_classes[select_model_class_idx]["models"]
selected_model = st.sidebar.selectbox(
    "Select Model", [model["name"] for model in models]
)

# Find the relevant model from the model class
idx = next(i for i, model in enumerate(models) if model["name"] == selected_model)

# For vision models, different pricing exists
if selected_model == "gpt-4-1106-vision-preview":
    vision_dict = models[idx].get("vision", {})
    if vision_dict:
        resolution = st.sidebar.selectbox("Select Resolution", vision_dict.keys())
        width = st.sidebar.number_input("Width", min_value=0, value=512, step=1)
        height = st.sidebar.number_input("Height", min_value=0, value=512, step=1)
        number_of_images = st.sidebar.number_input("Number of images", min_value=0, step=1)

        image_token_count = (
            calculate_vision_token_cost(width, height, detail=resolution)
            * number_of_images
        )
        st.sidebar.markdown(
            f"Number of tokens for {number_of_images} images: **{image_token_count}**"
        )

# Display input cost per {per_token} for the selected model in the sidebar
per_token = models[idx].get("per_token", 1)
input_cost_per_token = models[idx].get("input_cost", "NA")
output_cost_per_token = float(models[idx].get("output_cost", 0))
st.sidebar.markdown(
    f"""
    Input Cost per {per_token} tokens:   
    **${input_cost_per_token}** for **{selected_model}**
    """
)
st.sidebar.markdown(
    f"""
    Output Cost per {per_token} tokens:   
    **${output_cost_per_token}** for **{selected_model}**
    """
)

# Main content
st.title("Prompt To Price")

# Textbox for user input
user_input = st.sidebar.text_area("Enter your text:", key="input", height=200)

# Function to handle file upload
def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "json"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return " ".join(df.iloc[:, 0].astype(str).tolist())
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/json":
            return pd.read_json(uploaded_file).to_string()
    return None

# Handle file upload and update user input if a file is uploaded
uploaded_text = handle_file_upload()
if uploaded_text:
    user_input = uploaded_text

# Columns for buttons
col1, col2, *cols = st.columns(8)

# Button to calculate pricing
pricing_button = col1.button("Pricing")

# Calculate pricing if button is clicked or user input is provided
if pricing_button or user_input:
    with st.spinner():
        if selected_model.startswith("text-embedding"):
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model(selected_model)

        token_count = len(encoding.encode(user_input))
        output_token_count = st.sidebar.number_input("Number of output tokens", min_value=0, step=1)

        # Calculate total cost
        if selected_model == "gpt-4-1106-vision-preview":
            total_cost = (
                (token_count + image_token_count) * input_cost_per_token / per_token
                + output_token_count * output_cost_per_token / per_token
            )
            st.info(
                "Checkout https://platform.openai.com/docs/guides/vision/calculating-costs for more details"
            )
        else:
            total_cost = (
                token_count * input_cost_per_token / per_token
                + output_token_count * output_cost_per_token / per_token
            )

# Clear button to reset the text area
def clear_text():
    st.session_state["input"] = ""

col2.button("Clear", on_click=clear_text)

# Display the results in a table format
if user_input:
    data = {
        "Metric": ["Number of characters", "Number of Tokens", "Number of Output Tokens", "Input Cost per Token", "Output Cost per Token", "Total Cost"],
        "Value": [len(user_input), token_count, output_token_count, f"${input_cost_per_token}", f"${output_cost_per_token}", f"${total_cost:.7f}"]
    }

    if selected_model == "gpt-4-1106-vision-preview":
        data["Metric"].insert(3, "Number of Images Tokens")
        data["Value"].insert(3, image_token_count)

    result_df = pd.DataFrame(data).astype(str)
    st.table(result_df)
