import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import plotly.express as px
import json

@st.cache_resource
def load_model():
    # Using a publicly available Llama 2 model
    model_name = "NousResearch/Llama-2-7b-chat-hf"  # Changed model name
    
    # Load the model with M1-compatible settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=512):
    # Prepare the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def create_visualization(data, viz_type):
    if viz_type == "line":
        fig = px.line(data)
    elif viz_type == "bar":
        fig = px.bar(data)
    elif viz_type == "scatter":
        fig = px.scatter(data)
    elif viz_type == "pie":
        fig = px.pie(data)
    else:
        return None
    return fig

def main():
    st.set_page_config(page_title="Interactive LLM Analysis", layout="wide")
    st.title("Interactive Data Analysis with LLM")

    # Load the model
    model, tokenizer = load_model()

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        query = st.text_area("Enter your query (e.g., 'Create a table showing sales by region' or 'Show me a bar chart of monthly revenue')", 
                            height=100)
        
        if st.button("Generate Response"):
            if query:
                # Add specific instructions for structured output
                enhanced_prompt = f"""
                Analyze the following query and provide a response in JSON format with these fields:
                - explanation: A text explanation of the analysis
                - data: Any data to be displayed in a table (as a list of dictionaries)
                - visualization: Suggested visualization type (one of: line, bar, scatter, pie)
                - viz_data: Data for visualization

                Query: {query}

                Respond in valid JSON format only.
                """
                
                response = generate_response(enhanced_prompt, model, tokenizer)
                
                try:
                    # Try to parse the response as JSON
                    result = json.loads(response)
                    
                    # Display explanation
                    st.write("### Analysis")
                    st.write(result.get("explanation", "No explanation provided"))
                    
                    # Display table if data is present
                    if "data" in result and result["data"]:
                        st.write("### Data Table")
                        df = pd.DataFrame(result["data"])
                        st.dataframe(df)
                    
                    # Create visualization if specified
                    if "visualization" in result and "viz_data" in result:
                        st.write("### Visualization")
                        viz_data = pd.DataFrame(result["viz_data"])
                        fig = create_visualization(viz_data, result["visualization"])
                        if fig:
                            st.plotly_chart(fig)
                
                except json.JSONDecodeError:
                    st.error("Could not parse the model's response into a valid format")
                    st.text(response)

    with col2:
        st.write("### Query Examples")
        st.write("""
        Try these example queries:
        - Show me sales data by region
        - Create a pie chart of market share
        - Display monthly revenue trends
        - Compare product performance
        """)
        
        st.write("### Tips")
        st.write("""
        - Be specific about the type of visualization you want
        - Mention the metrics you're interested in
        - Specify any time periods or categories
        """)

if __name__ == "__main__":
    main() 