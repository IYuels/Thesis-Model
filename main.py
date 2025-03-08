import os
import re
import json
import torch
import torch.nn as nn
import fasttext
import fasttext.util
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse



# Define the model architecture (same as in your training code)
class HybridCNNBiLSTM(nn.Module):
    def __init__(self, embedding_dim, max_length, num_classes, dropout_rate=0.5):
        super(HybridCNNBiLSTM, self).__init__()

        # CNN part with multiple filter sizes
        self.filter_sizes = [2, 3, 4, 5]
        self.num_filters = 64

        # Conv layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=self.num_filters,
                     kernel_size=fs)
            for fs in self.filter_sizes
        ])

        # BiLSTM part
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_size=64,
                           bidirectional=True,
                           batch_first=True)

        # Calculated size of concatenated features
        cnn_output_dim = self.num_filters * len(self.filter_sizes)
        lstm_output_dim = 64 * 2  # bidirectional = 2 * hidden_size

        # Dense layers
        self.fc1 = nn.Linear(cnn_output_dim + lstm_output_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        # Input x shape: [batch_size, seq_len, embedding_dim]

        # For CNN: convert to [batch_size, embedding_dim, seq_len]
        x_conv = x.permute(0, 2, 1)

        # Apply CNN layers and max-over-time pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution
            conv_out = conv(x_conv)
            # Apply ReLU
            conv_out = self.relu(conv_out)
            # Apply max pooling over time
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        # Concatenate all CNN outputs
        cnn_features = torch.cat(conv_outputs, dim=1)

        # For BiLSTM
        # Pack padded sequence for efficient LSTM computation
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Apply BiLSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Get the output for the last non-padded element in each sequence
        batch_size = lstm_out.size(0)

        # Create a tensor of indices for the last element of each sequence
        idx = (lengths - 1).view(-1, 1).expand(batch_size, lstm_out.size(2)).unsqueeze(1)

        # Gather the last output for each sequence
        lstm_features = lstm_out.gather(1, idx).squeeze(1)

        # Concatenate CNN and BiLSTM features
        combined = torch.cat([cnn_features, lstm_features], dim=1)

        # Apply dense layers
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

# Text cleaning function
def clean_text(text):
    """
    Clean the text using regex
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[ \n]+', ' ', text)
        text = re.sub(r"[=%;]", "", text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    else:
        return ""

# Define Pydantic models for request and response
class TextRequest(BaseModel):
    text: str

class TextsRequest(BaseModel):
    texts: List[str]

class ToxicityResult(BaseModel):
    TOXICITY: Dict[str, Any]
    SEVERE_TOXICITY: Dict[str, Any]
    INSULT: Dict[str, Any]
    PROFANITY: Dict[str, Any]
    IDENTITY_ATTACK: Dict[str, Any]
    THREAT: Dict[str, Any]
    NOT_TOXIC: Dict[str, Any]

class ToxicityResponse(BaseModel):
    results: Dict[str, ToxicityResult]

# Initialize FastAPI app
app = FastAPI(title="Toxicity Classification API",
              description="API for classifying text toxicity using a hybrid CNN-BiLSTM model",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for the model and embeddings
model = None
fasttext_model = None
max_length = None
device = None
label_names = ['TOXICITY', 'SEVERE_TOXICITY', 'INSULT', 'PROFANITY', 'IDENTITY_ATTACK', 'THREAT', 'NOT_TOXIC']

@app.get("/")
async def root():
    """
    Redirect root to docs
    """
    return RedirectResponse(url="/docs")

@app.on_event("startup")
async def startup_event():
    """
    Load model and fastText embeddings on startup
    """
    global model, fasttext_model, max_length, device
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the model
    model_path = os.environ.get("MODEL_PATH", "model/hybrid_textcnn_bilstm_model 7 new.pt")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create a new model with the same architecture
        model = HybridCNNBiLSTM(
            embedding_dim=checkpoint['embedding_dim'],
            max_length=checkpoint['max_length'],
            num_classes=checkpoint['num_classes']
        ).to(device)
        
        # Load the trained parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        max_length = checkpoint['max_length']
        
        # Load fastText embeddings
        fasttext_path = os.environ.get("FASTTEXT_PATH", "model/fasttext_model2.bin")
        fasttext_model = fasttext.load_model(fasttext_path)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict_single_text(text):
    """
    Predict toxicity for a single input text
    """
    # Clean the text
    cleaned_text = clean_text(text)

    # Tokenize
    tokens = cleaned_text.split()

    # Get length
    length = min(len(tokens), max_length)

    # Truncate if needed
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    # Get fastText embeddings for each word
    word_embeddings = []
    for word in tokens:
        embedding = torch.tensor(fasttext_model.get_word_vector(word))
        word_embeddings.append(embedding)

    # If the text is empty, add a zero vector
    if len(word_embeddings) == 0:
        embedding_dim = fasttext_model.get_dimension()
        word_embeddings.append(torch.zeros(embedding_dim))
        length = 1

    # Convert to tensor
    embeddings = torch.stack(word_embeddings)

    # Pad if needed
    if embeddings.size(0) < max_length:
        padding = torch.zeros(max_length - embeddings.size(0), embeddings.size(1))
        embeddings = torch.cat([embeddings, padding], dim=0)

    # Create batch input (batch size of 1)
    batch_embeddings = embeddings.unsqueeze(0).to(device)
    batch_lengths = torch.tensor([length]).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(batch_embeddings, batch_lengths)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = (probabilities > 0.5).astype(int)

    # Create results dictionary
    results = {}
    for i, label in enumerate(label_names):
        results[label] = {
            'prediction': bool(predictions[i]),
            'probability': float(probabilities[i])
        }

    return results

@app.post("/predict", response_model=ToxicityResult)
async def predict(request: TextRequest):
    """
    Classify a single text for toxicity
    """
    if model is None or fasttext_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predict_single_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=ToxicityResponse)
async def predict_batch(request: TextsRequest):
    """
    Classify multiple texts for toxicity
    """
    if model is None or fasttext_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = {}
        for i, text in enumerate(request.texts):
            results[str(i)] = predict_single_text(text)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if model is None or fasttext_model is None:
        return {"status": "not_ready", "message": "Model not loaded"}
    return {"status": "ready", "device": device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)