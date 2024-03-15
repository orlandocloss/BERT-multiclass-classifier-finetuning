import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = '/mnt/sda1/local-working/action-classifier/action_classifier_model'  # Update this path
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text, threshold=0.5):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.sigmoid(outputs.logits).squeeze()
    
    predictions = (probabilities >= threshold).long()
    
    return predictions.cpu().numpy()

text_input = input("Enter your text: ")  # Accept text input from user
predictions = predict(text_input, threshold=0.2)  # You can adjust the threshold

print("Predictions:", predictions)
