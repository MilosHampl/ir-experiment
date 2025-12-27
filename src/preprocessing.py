import re
import string
from typing import List

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "in", "of", "on", "to", "with", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"
}

def preprocess_text(text: str) -> List[str]:
    """
    Preprocesses text for sparse retrieval:
    1. Lowercasing
    2. Removing punctuation
    3. Tokenization (splitting by whitespace)
    4. Stop-word removal
    
    Args:
        text: Input string.
        
    Returns:
        List of tokens.
    """
    if not text:
        return []
        
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize by whitespace
    tokens = text.split()
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    return tokens

