import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_dir = '/mnt/sda1/local-working/action-classifier/action_classifier_model'  # Update this path
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Ensure model is in evaluation mode
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to process a single text input and return the prediction
def predict(text, threshold=0.5):
    # Tokenize text input
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities
    probabilities = torch.sigmoid(outputs.logits).squeeze()
    
    # Apply threshold to get binary predictions
    predictions = (probabilities >= threshold).long()
    
    return predictions.cpu().numpy()

# Example usage
text_input = input("Enter your text: ")  # Accept text input from user
predictions = predict(text_input, threshold=0.2)  # You can adjust the threshold

# For demonstration, print the binary predictions array
print("Predictions:", predictions)

# Note: You'll need to map these binary predictions back to your label names based on their positions.
