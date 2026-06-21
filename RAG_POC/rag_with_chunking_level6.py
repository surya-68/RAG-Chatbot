#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdfplumber
import re
import nltk
import networkx as nx
import faiss
import requests
import json
import numpy as np
import networkx as nx
import time
import os
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import find


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sacrebleu import corpus_bleu
from rouge import Rouge
from transformers import pipeline


# Download necessary NLTK data (SAFE & REQUIRED)
# nltk.download('punkt')
# nltk.download('punkt_tab')   # ⭐ THIS WAS MISSING
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')



# In[85]:


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.

    Args:
    pdf_path (str): The file path to the PDF.

    Returns:
    str: Extracted text from the PDF. Returns an empty string if an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: The file {pdf_path} does not exist.")
        return ''
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if the page contains text
                    text += page_text
                else:
                    print(f"Warning: Page {page.page_number} in {pdf_path} contains no extractable text.")
        return text
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")
        return ''



# PDF Paths
pdf_path4 = r"E:\Project\RAG_POC - Copy (2)\RAG_POC\HR_Docs\BCA_docs_for_RAG.pdf"



# Extract Text

pdf_text4 = extract_text_from_pdf(pdf_path4)

# Trimming the text from the first PDF
# if pdf_text1:
#     pdf_text1 = pdf_text1[15529:89000]

# Concatenating all the extracted texts
# pdf_text = pdf_text1 + pdf_text2 + pdf_text3 + pdf_text4
pdf_text = pdf_text4


# In[86]:


def clean_text(text):
    try:
        # Ensure NLTK resources are available
        find('tokenizers/punkt')
        find('corpora/stopwords')
    except LookupError:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # Remove HTML tags
    text = re.sub(r"<[^>]*>", "", text)

    # Preserve ranges and special cases
    text = re.sub(r"(\d+\s*[-–]\s*\d+\s*(years|months|days|hrs|minutes|seconds)?)", lambda x: x.group(0), text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+\+\s*(years|months|days|hrs|minutes|seconds)?)", lambda x: x.group(0), text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+,*\d*/-)", lambda x: x.group(0).replace(',', ''), text)  # Standardize numeric values

    # Remove numerical and Roman numeral pointers
    text = re.sub(r"\b\d+\.\s+|\b(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\.\s+", "", text, flags=re.IGNORECASE)

    # Remove special characters except for decimals, hyphens, plus signs, slashes, '@'
    text = re.sub(r"[^a-zA-Z0-9\s.\-+/@]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)

    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# Example usage
cleaned_text = clean_text(pdf_text)


# In[55]:


def sliding_window_chunk(text, chunk_size=100, overlap_size=10):
    """
    Splits the given text into chunks using a sliding window approach.

    Args:
    text (str): The input text to be split into chunks.
    chunk_size (int): The size of each chunk (number of words). Must be greater than 0.
    overlap_size (int): The number of overlapping words between consecutive chunks. Cannot be negative.

    Returns:
    List[str]: A list of text chunks.
    
    Raises:
    ValueError: If chunk_size is less than or equal to 0, or if overlap_size is negative.
    ValueError: If chunk_size is less than or equal to overlap_size.
    """
    words = text.split()
    chunks = []

    # Validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative.")
    if chunk_size <= overlap_size:
        raise ValueError("chunk_size must be greater than overlap_size.")

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start += chunk_size - overlap_size
        if start >= len(words):
            break

    return chunks

    '''
def test_chunk_sizes(text, chunk_sizes, overlap_size=10):
    results = {}
    
    for chunk_size in chunk_sizes:
        print(f"Testing chunk size: {chunk_size}")
        
        # Measure time for processing
        start_time = time.time()
        chunks = sliding_window_chunk(text, chunk_size, overlap_size)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        results[chunk_size] = (elapsed_time, len(chunks))
        
        print(f"Chunk size {chunk_size} took {elapsed_time:.4f} seconds. Number of chunks: {len(chunks)}")
    
    return results
text = cleaned_text
chunk_sizes = [100, 250, 500, 750, 1000]  # Example chunk sizes
results = test_chunk_sizes(text, chunk_sizes)

# Display results
print("\nPerformance Results:")
for chunk_size, (elapsed_time, num_chunks) in results.items():
    print(f"Chunk size: {chunk_size}, Time: {elapsed_time:.4f} seconds, Number of chunks: {num_chunks}")
    Testing chunk size: 100
Chunk size 100 took 0.0010 seconds. Number of chunks: 72
Testing chunk size: 250
Chunk size 250 took 0.0000 seconds. Number of chunks: 27
Testing chunk size: 500
Chunk size 500 took 0.0000 seconds. Number of chunks: 14
Testing chunk size: 750
Chunk size 750 took 0.0000 seconds. Number of chunks: 9
Testing chunk size: 1000
Chunk size 1000 took 0.0010 seconds. Number of chunks: 7

Performance Results:
Chunk size: 100, Time: 0.0010 seconds, Number of chunks: 72
Chunk size: 250, Time: 0.0000 seconds, Number of chunks: 27
Chunk size: 500, Time: 0.0000 seconds, Number of chunks: 14
Chunk size: 750, Time: 0.0000 seconds, Number of chunks: 9
Chunk size: 1000, Time: 0.0010 seconds, Number of chunks: 7 '''


# In[8]:


def combine_chunks(chunks):
    """
    Combines a list of text chunks into a single string with each chunk separated by a newline.

    Args:
    chunks (List[str]): A list of text chunks to be combined.

    Returns:
    str: A single string with all chunks combined, each separated by a newline.
    """
    if not isinstance(chunks, list):
        raise TypeError("Input must be a list of strings.")
    
    return "\n".join(chunks)


# In[57]:


def create_embeddings(chunks, model, batch_size=128):
    """
    Creates embeddings for a list of text chunks using a specified model.
    Args:
    chunks (List[str]): A list of text chunks to generate embeddings for.
    model (SentenceTransformer): A pre-trained SentenceTransformer model to generate embeddings.
    batch_size (int, optional): The size of each batch to process at once. Defaults to 128.

    Returns:
    np.ndarray: A 2D numpy array containing embeddings for all chunks.

    Raises:
    ValueError: If the list of chunks is empty.
    RuntimeError: If there is an error during the embedding generation process.
    TypeError: If the chunks are not provided as a list or the model is not a SentenceTransformer instance.
    """
    # Validate input types
    if not isinstance(chunks, list):
        raise TypeError("The chunks should be provided as a list.")
    if not isinstance(model, SentenceTransformer):
        raise TypeError("The model should be an instance of SentenceTransformer.")
    
    if not chunks:
        raise ValueError("The list of chunks is empty.")
    
    chunk_embeddings = []
    
    # Process chunks in batches
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]
        
        try:
            # Generate embeddings for the current batch using SentenceTransformer's batch processing
            batch_embeddings = model.encode(batch_chunks, convert_to_numpy=True)
            chunk_embeddings.append(batch_embeddings)
        except Exception as e:
            raise RuntimeError(f"Error while generating embeddings for batch {start // batch_size}: {e}")
    
    # Concatenate all batch embeddings into a single array
    return np.vstack(chunk_embeddings)


# In[59]:


def create_faiss_index(chunk_embeddings):
    """
    Creates a FAISS index for fast similarity search using L2 distance.

    Args:
    chunk_embeddings (np.ndarray): A 2D numpy array where each row is an embedding vector.

    Returns:
    faiss.IndexFlatL2: A FAISS index built using the provided embeddings.

    Raises:
    ValueError: If chunk_embeddings is not a 2D numpy array.
    RuntimeError: If the FAISS index cannot be created for any reason.
    """
    
    # Validate input type and dimensions
    if not isinstance(chunk_embeddings, np.ndarray):
        raise ValueError("chunk_embeddings must be a numpy array.")
    if chunk_embeddings.ndim != 2:
        raise ValueError("chunk_embeddings must be a 2D numpy array with shape (n_samples, embedding_dim).")
    
    try:
        # Determine the dimensionality of the embeddings
        embedding_dim = chunk_embeddings.shape[1]
        
        # Create a FAISS index using L2 (Euclidean) distance
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to the index
        index.add(chunk_embeddings)
        
    except Exception as e:
        raise RuntimeError(f"Error while creating FAISS index: {e}")
    
    return index


# In[61]:


# Initialize the SentenceTransformer model
try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Generate text chunks using the sliding window approach
try:
    chunks = sliding_window_chunk(cleaned_text)
except Exception as e:
    raise RuntimeError(f"Error during text chunking: {e}")

# Create embeddings for the chunks
try:
    chunk_embeddings = create_embeddings(chunks, model)
except Exception as e:
    raise RuntimeError(f"Error creating embeddings: {e}")

# Create a FAISS index from the embeddings
try:
    index = create_faiss_index(chunk_embeddings)
except Exception as e:
    raise RuntimeError(f"Error creating FAISS index: {e}")


# In[63]:


def retrieval(query, model, index, chunks, top_k=6):
    # Validate inputs
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Chunks must be a non-empty list of strings.")
    
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    
    try:
        # Encode the query to get its embedding
        query_embedding = model.encode([query])
    except Exception as e:
        raise RuntimeError(f"Error encoding the query: {e}")

    try:
        # Search the FAISS index
        distances, indices = index.search(query_embedding, k=top_k)
    except Exception as e:
        raise RuntimeError(f"Error searching the FAISS index: {e}")

    try:
        # Calculate relevance scores using cosine similarity
        chunk_embeddings = np.array([model.encode(chunk) for chunk in chunks], dtype=np.float32)
        scores = cosine_similarity(query_embedding, chunk_embeddings[indices[0]])
    except Exception as e:
        raise RuntimeError(f"Error calculating cosine similarity: {e}")

    try:
        # Retrieve the top chunks based on sorted scores
        sorted_indices = np.argsort(scores[0])[::-1]  # Sort indices by score in descending order
        top_indices = indices[0][sorted_indices]  # Retrieve the indices of the top chunks
        
        # Select the top chunks based on sorted indices
        retrieved_chunks = [chunks[i] for i in top_indices[:top_k]]
    except Exception as e:
        raise RuntimeError(f"Error retrieving top chunks: {e}")

    return retrieved_chunks


# In[65]:


def summarize_text(text, max_length=180, min_length=40):
    # Validate input text
    if not text or not isinstance(text, str):
        raise ValueError("The input text must be a non-empty string.")
    
    if not isinstance(max_length, int) or not isinstance(min_length, int):
        raise ValueError("max_length and min_length must be integers.")
    
    if min_length > max_length:
        raise ValueError("min_length cannot be greater than max_length.")
    
    try:
        # Initialize BART model and tokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    except Exception as e:
        raise RuntimeError(f"Error loading BART model or tokenizer: {e}")
    
    try:
        # Tokenize and summarize
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=1.0, num_beams=4, early_stopping=True)
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Error during summarization: {e}")
    
    return summary


# In[67]:


def get_llm_response(prompt, base_url="http://ollama:11434", model="llama3", temperature=0.4):
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string.")
    
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        'model': model,
        'prompt': prompt,
        'temperature': temperature
    }
    
    try:
        response = requests.post(base_url + '/api/generate', headers=headers, data=json.dumps(data), timeout=300)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if response.status_code == 200:
        try:
            response_text = response.text.strip()
            json_objects = response_text.split('\n')
            full_response = ''.join(json.loads(obj)['response'] for obj in json_objects)
            return full_response
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            print(f"Response text: {response.text}")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


# In[69]:


def generate_chatbot_response(user_query, retrieved_chunks):
    try:
        # Validate retrieved chunks
        if not retrieved_chunks:
            raise ValueError("Retrieved chunks are empty.")
        
        # Combine chunks
        context_text = combine_chunks(retrieved_chunks)
        
        if not context_text:
            raise ValueError("Context text is empty after combining chunks.")
    
        # Summarize the context text
        summary = summarize_text(context_text)
        if not summary:
            raise ValueError("Summary is empty after summarizing context text.")
    
        # Formulate the prompt
        prompt = f"""
        Context:
        {context_text}
        
        Summary:
        {summary}
        
        User Query:
        {user_query}
        
        Please provide an accurate response based on the context and summary above. Avoid introductory phrases such as "According to the context" or similar, and do not use phrases like "is not mentioned in the document or text."
        """
    
        # Get the response from the LLM
        response = get_llm_response(prompt)
        if not response:
            raise ValueError("Response from the LLM is empty.")
        return response
    
    except Exception as e:
        # Log the error (if logging is enabled)
        # logging.error(f"Error in generating chatbot response: {e}")
        print(f"Error in generating chatbot response: {e}")
        return "Sorry, I'm having trouble generating a response right now."


# In[73]:


def preprocess_text(text):
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Clean tokens
    cleaned_tokens = [re.sub(r"[^a-zA-Z0-9]", "", token.lower()) for token in tokens]
    
    # Remove empty tokens
    return [token for token in cleaned_tokens if token]


# In[75]:


def evaluate_response(generated_text, reference_text):
    # Validate input types
    if not isinstance(generated_text, str) or not isinstance(reference_text, str):
        raise TypeError("Both generated_text and reference_text must be strings.")
    
    # Check for empty strings
    if not generated_text.strip() or not reference_text.strip():
        raise ValueError("Both generated_text and reference_text must be non-empty strings.")
    
    # Clean and tokenize reference and generated text
    cleaned_reference_tokens = preprocess_text(reference_text)
    cleaned_generated_tokens = preprocess_text(generated_text)

    # Join tokens back into strings for ROUGE
    cleaned_reference_text = ' '.join(cleaned_reference_tokens)
    cleaned_generated_text = ' '.join(cleaned_generated_tokens)

    # Calculate ROUGE scores
    try:
        rouge = Rouge()
        rouge_scores = rouge.get_scores(cleaned_generated_text, cleaned_reference_text, avg=True)
    except Exception as e:
        print(f"Error during ROUGE score calculation: {e}")
        rouge_scores = {}

    return rouge_scores


# In[77]:


def get_result(query):
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")

    # Try to retrieve relevant chunks
    try:
        retrieved_chunks = retrieval(query, model, index, chunks)
        if not retrieved_chunks:
            raise ValueError("No relevant chunks could be retrieved for the query.")
    except Exception as e:
        return f"Error during retrieval: {e}"

    # Try to generate a response
    try:
        result = generate_chatbot_response(query, retrieved_chunks)
        if not result:
            raise ValueError("The response generated is empty.")
    except Exception as e:
        return f"Error during response generation: {e}"

    return result


