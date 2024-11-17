# TeeDeeAI
FuncTee &amp; DataDee

# FuncTee
FuncTee is the builder’s hub, where data scientists and AI developers can create, test, and upload their AI models. With intuitive tools and support, FuncTee makes it easy for creators to transform their expertise into functional, deployable models ready for integration.

# DataDee
DataDee is the marketplace that showcases all the models uploaded on FuncTee. Users can browse, evaluate, and purchase AI solutions tailored to their needs. With a diverse catalog of innovative models, DataDee ensures accessibility and flexibility for businesses and individuals looking to leverage the power of AI.

---

## Workflow

### For Creators (Using FuncTee):

Build and upload models via FuncTee’s tools.
Configure pricing, licensing, and visibility for their models.
Automatically publish to DataDee for discovery and monetization.

### For Users (Using DataDee):

Browse, evaluate, and purchase models from the marketplace.
Access details about the models’ functionality and pricing.
Download or integrate models seamlessly with their applications.

### Key Features:

Seamless Integration: Models flow from FuncTee to DataDee automatically.
Revenue Sharing: Transparent revenue split between creators and the platform.
Community Engagement: Offer forums, tutorials, and ratings to build trust and collaboration.
Analytics & Insights: Provide data for creators on model performance and user engagement.


## How to Run


1. Run the TEE Remote Attestation Simulator:

```bash
docker run --rm -p 8090:8090 phalanetwork/tappd-simulator:latest
```

2. Use uv to sync the project and run the app
```bash
uv venv
source .venv/bin/activate
uv sync 
# uv pip install -r requirements.txt
python FuncTee/app.py
```

FuncTee will be available at http://localhost:8000/