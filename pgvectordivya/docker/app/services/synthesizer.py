#synthesizer.py
import pandas as pd
from typing import List, Dict, Optional, Union
import logging
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

class Synthesizer:
    # Initialize the model and tokenizer
    try:
        model_name = "BAAI/bge-large-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        model = None
        tokenizer = None

    @classmethod
    def get_bert_embeddings(cls, text: str) -> np.ndarray:
        """Generate embeddings for input text using model.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embeddings
        """
        # Tokenize and encode text
        inputs = cls.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(cls.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = cls.model(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            embeddings = outputs.logits.mean(dim=1).cpu().numpy()

            
        return embeddings[0]  # Return the first (and only) embedding

    @classmethod
    def cosine_similarity(cls, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    @classmethod
    def generate_response(
        cls,
        question: str,
        context: pd.DataFrame,
    ) -> Dict:
        """Generate a response using the context provided.
        
        Args:
            question: The user's question
            context: DataFrame containing the search results
            
        Returns:
            Dictionary containing the answer, thought process, and context sufficiency
        """
        if cls.model is None or cls.tokenizer is None:
            return {
                "answer": "Model not initialized properly.",
                "thought_process": ["Error: Model loading failed"],
                "enough_context": False
            }

        try:
            # Get question embedding
            question_embedding = cls.get_bert_embeddings(question)
            
            # Process and rank context entries
            context_entries = []
            for _, row in context.iterrows():
                content = row['content']
                content_embedding = cls.get_bert_embeddings(content)
                similarity = cls.cosine_similarity(question_embedding, content_embedding)
                context_entries.append({
                    'content': content,
                    'similarity': similarity,
                    'document_name': row.get('Documentname', 'Unknown'),
                    'created_at': row.get('created_at', 'Unknown'),
                    'governing_law': row.get('Governing Law', 'Unknown'),
                })
            
            # Sort by similarity
            context_entries.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Generate structured response
            if not context_entries:
                return {
                    "answer": "No relevant context found to answer the question.",
                    "thought_process": ["No matching documents found in the context"],
                    "enough_context": False
                }
            
            # Use the most relevant context to construct the answer
            best_match = context_entries[0]
            
            # Determine if we have enough context based on similarity threshold
            enough_context = best_match['similarity'] > 0.5
            
            if enough_context:
                # Construct response
                answer = (
                    f"Based on the document '{best_match['document_name']}' "
                    f"(created {best_match['created_at']}), "
                    f"the relevant information is: {best_match['content']}"
                )
            else:
                print("No relevant context found to answer the question.")
            thought_process = [
                f"Analyzed similarity between question and {len(context_entries)} documents",
                f"Best matching document has similarity score of {best_match['similarity']:.2f}",
                "Generated response using the most relevant context",
                f"Context source: {best_match['document_name']}"
            ]
            
            return {
                "answer": answer,
                "thought_process": thought_process,
                "enough_context": enough_context
            }
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Sorry, I couldn't generate a response at this time.",
                "thought_process": ["Error occurred during processing"],
                "enough_context": False
            }

    @staticmethod
    def dataframe_to_json(context: pd.DataFrame) -> str:
        """Convert DataFrame to JSON string, keeping only relevant columns.
        
        Args:
            context: DataFrame containing search results
            
        Returns:
            JSON string representation of the context
        """
        columns_to_keep = ['content', 'Documentname', 'created_at', 'distance']
        available_columns = [col for col in columns_to_keep if col in context.columns]
        
        if not available_columns:
            return "[]"
            
        return context[available_columns].to_json(orient="records", indent=2)