import os
import asyncio
import uvicorn
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from dstack_sdk import AsyncTappdClient, DeriveKeyResponse, TdxQuoteResponse

def create_app():
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize ML model
    model = LinearRegression()
    
    # # Direct API endpoints
    # @app.get("/api/derivekey/{subject}")
    # async def derivekey(subject: str):
    #     client = AsyncTappdClient()
    #     deriveKey = await client.derive_key('/', subject)
    #     assert isinstance(deriveKey, DeriveKeyResponse)
    #     asBytes = deriveKey.toBytes()
    #     assert isinstance(asBytes, bytes)
    #     limitedSize = deriveKey.toBytes(32)
    #     return {"deriveKey": asBytes.hex(), "derive_32bytes": limitedSize.hex()}
        
    # @app.get("/api/tdxquote/{subject}")
    # async def tdxquote(subject: str):
    #     client = AsyncTappdClient()
    #     tdxQuote = await client.tdx_quote(subject)
    #     assert isinstance(tdxQuote, TdxQuoteResponse)
    #     return {"tdxQuote": str(tdxQuote)}
    
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
        
    # Gradio Interface
    def create_interface():
        with gr.Blocks(title="FuncTee") as interface:
            gr.Markdown("""
                # FuncTee
                    
                FuncTee is the builder's hub, where data scientists and AI developers 
                can create, test, and upload their AI models. With intuitive tools 
                and support, FuncTee makes it easy for creators to transform their 
                expertise into functional, deployable models ready for integration.
            """)
            
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
                    path_input = gr.Textbox(label="Path", placeholder="/")
                    test_param_derive = gr.Textbox(label="Test Parameter", placeholder="test")
                    derive_button = gr.Button("Get Derive Key")
                    derive_output = gr.Textbox(label="Derive Key Result")
                
                derive_button.click(
                    fn=derive_key_wrapper,
                    inputs=[path_input, test_param_derive],
                    outputs=derive_output
                )
            
            with gr.Tab("TDX Quote"):
                with gr.Column():
                    tdx_input = gr.Textbox(label="Subject", placeholder="test")
                    tdx_button = gr.Button("Get TDX Quote")
                    tdx_output = gr.Textbox(label="TDX Quote Result")
                
                tdx_button.click(
                    tdx_quote_wrapper,
                    inputs=[tdx_input],
                    outputs=tdx_output
                )
        
        return interface

    # Wrapper for async functions to work with Gradio
    def derive_key_wrapper(path, test_param):
        return asyncio.run(get_derive_key(path, test_param))
    
    def tdx_quote_wrapper(test_param):
        return asyncio.run(get_tdx_quote(test_param))
    
    # Async functions for Gradio
    async def get_derive_key(path, test_param):
        client = AsyncTappdClient(endpoint="http://localhost:8090")  # or whatever your Tappd server URL is
        deriveKey = await client.derive_key(path, test_param)
        assert isinstance(deriveKey, DeriveKeyResponse)
        asBytes = deriveKey.toBytes()
        limitedSize = deriveKey.toBytes(32)
        return f"DeriveKey: {asBytes.hex()}\nDerive 32bytes: {limitedSize.hex()}"

    async def get_tdx_quote(test_param):
        client = AsyncTappdClient(endpoint="http://localhost:8090")  # or whatever your Tappd server URL is
        tdxQuote = await client.tdx_quote(test_param)
        assert isinstance(tdxQuote, TdxQuoteResponse)
        return f"TDX Quote: {tdxQuote}"
    
    # Create and mount Gradio interface
    interface = create_interface()
    app = gr.mount_gradio_app(app, interface, path="/")
    
    return app

def main():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()