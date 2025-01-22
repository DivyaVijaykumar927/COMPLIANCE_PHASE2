import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your cleaned CSV file
df = pd.read_csv('cleaned_file.csv')

# Load the LegalBERT model and tokenizer
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to process each row using LegalBERT
def process_row(row):
    prompt = f"""
    Based on the following contract details, determine:
    1. A major governing law applicable (e.g., 'Contract Act', 'Commercial Code', 'Consumer Protection Act').
    2. Any discrepancy if apparent in the provided details.

    Details:
    Category: {row['Category']}
    Document Name: {row['Document Name']}
    Parties: {row['Parties']}
    Agreement Date: {row['Agreement Date']}
    Effective Date: {row['Effective Date']}
    Expiration Date: {row['Expiration Date']}
    Renewal Term: {row['Renewal Term']}
    Governing Law: {row['Governing Law']}
    Notice To Terminate Renewal: {row['Notice To Terminate Renewal']}
    Exclusivity: {row['Exclusivity']}
    Post-Termination Services: {row['Post-Termination Services']}

    Provide your response in the following format:
    Exact_Law: <'Contract Act', 'Commercial Code', 'Consumer Protection Act', 'Civil Law'> or more than one
    Discrepancy: <Briefly describe the discrepancy, or state 'None' if no discrepancy exists>
    """
    
    try:
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract logits and determine labels (This assumes your model is fine-tuned for this task)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        # Map the predicted label to a meaningful value (adjust this mapping based on your model's output)
        exact_law = "Contract Act" if predicted_label == 0 else "Commercial Code"  # Example mapping
        discrepancy = "None" if predicted_label == 0 else "Potential mismatch found"
        
        return pd.Series([exact_law, discrepancy])
    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series(['Error', 'Error'])

# Apply the function to each row in the DataFrame
df[['Exact_Law', 'Discrepancy']] = df.apply(process_row, axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv('processed_file.csv', index=False)

print("Processing complete! The updated file is saved as 'processed_file.csv'")
