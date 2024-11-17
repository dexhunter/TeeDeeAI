import os
from dstack_sdk import AsyncTappdClient, DeriveKeyResponse, TdxQuoteResponse
from fastapi import FastAPI
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize a simple ML model
model = LinearRegression()

# Function for derivekey
async def get_derive_key(path, test_param):
    client = AsyncTappdClient()
    deriveKey = await client.derive_key(path, test_param)
    assert isinstance(deriveKey, DeriveKeyResponse)
    asBytes = deriveKey.toBytes()
    limitedSize = deriveKey.toBytes(32)
    return f"DeriveKey: {asBytes.hex()}\nDerive 32bytes: {limitedSize.hex()}"

# Function for tdxquote
async def get_tdx_quote(test_param):
    client = AsyncTappdClient()
    tdxQuote = await client.tdx_quote(test_param)
    assert isinstance(tdxQuote, TdxQuoteResponse)
    return f"TDX Quote: {tdxQuote}"

# Wrapper for async functions to work with Gradio
def derive_key_wrapper(path, test_param):
    return asyncio.run(get_derive_key(path, test_param))

def tdx_quote_wrapper(test_param):
    return asyncio.run(get_tdx_quote(test_param))

# ML functions
def train_model(file, target_column):
    try:
        df = pd.read_csv(file.name)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        model.fit(X, y)
        return f"Model trained successfully! Features: {X.columns.tolist()}"
    except Exception as e:
        return f"Error training model: {str(e)}"

def predict(file):
    try:
        test_df = pd.read_csv(file.name)
        predictions = model.predict(test_df)
        return f"Predictions: {predictions.tolist()}"
    except Exception as e:
        return f"Error making predictions: {str(e)}"

# Create Gradio interface with tabs
with gr.Blocks(title="ML Model and API Testing Interface") as interface:
    gr.Markdown("# ML Model and API Testing Interface")
    
    with gr.Tab("ML Model"):
        with gr.Row():
            with gr.Column():
                train_file = gr.File(label="Upload Training Data (CSV)")
                target_col = gr.Textbox(label="Target Column Name")
                train_button = gr.Button("Train Model")
                training_output = gr.Textbox(label="Training Result")
            
            with gr.Column():
                test_file = gr.File(label="Upload Test Data (CSV)")
                predict_button = gr.Button("Make Predictions")
                prediction_output = gr.Textbox(label="Predictions")
        
        train_button.click(
            train_model,
            inputs=[train_file, target_col],
            outputs=training_output
        )
        predict_button.click(
            predict,
            inputs=[test_file],
            outputs=prediction_output
        )
    
    with gr.Tab("Derive Key"):
        with gr.Column():
            path_input = gr.Textbox(label="Path", value="/")
            test_param_derive = gr.Textbox(label="Test Parameter", value="test")
            derive_button = gr.Button("Get Derive Key")
            derive_output = gr.Textbox(label="Derive Key Result")
        
        derive_button.click(
            derive_key_wrapper,
            inputs=[path_input, test_param_derive],
            outputs=derive_output
        )
    
    with gr.Tab("TDX Quote"):
        with gr.Column():
            test_param_tdx = gr.Textbox(label="Test Parameter", value="test")
            tdx_button = gr.Button("Get TDX Quote")
            tdx_output = gr.Textbox(label="TDX Quote Result")
        
        tdx_button.click(
            tdx_quote_wrapper,
            inputs=[test_param_tdx],
            outputs=tdx_output
        )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")