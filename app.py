import yaml
import tiktoken
import streamlit as st
import pandas as pd

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

# Textbox for user input
user_input = st.sidebar.text_area("Enter your text to calculate token:", key="input", height=200)

# Number input for output tokens
output_token_count = st.sidebar.number_input("Number of output tokens", min_value=0, step=1)

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
        results = []
        for model_class in model_classes:
            for model in model_class["models"]:
                if model["name"].startswith("text-embedding"):
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    try:
                        encoding = tiktoken.encoding_for_model(model["name"])
                    except KeyError:
                        encoding = tiktoken.get_encoding("cl100k_base")

                token_count = len(encoding.encode(user_input))
                per_token = model.get("per_token", 1)
                input_cost_per_token = model.get("input_cost", "NA")
                output_cost_per_token = float(model.get("output_cost", 0))

                if model["name"] == "gpt-4-1106-vision-preview":
                    total_cost = (
                        (token_count) * input_cost_per_token / per_token
                        + output_token_count * output_cost_per_token / per_token
                    )
                else:
                    total_cost = (
                        token_count * input_cost_per_token / per_token
                        + output_token_count * output_cost_per_token / per_token
                    )

                results.append({
                    "Model Class": model_class["name"],
                    "Model": model["name"],
                    "Number of Characters": len(user_input),
                    "Number of Tokens": token_count,
                    "Number of Output Tokens": output_token_count,
                    "Input Cost per Token": f"${input_cost_per_token}",
                    "Output Cost per Token": f"${output_cost_per_token}",
                    "Total Cost": f"${total_cost:.7f}"
                })

        result_df = pd.DataFrame(results)
        st.table(result_df)

# Clear button to reset the text area
def clear_text():
    st.session_state["input"] = ""

col2.button("Clear", on_click=clear_text)
