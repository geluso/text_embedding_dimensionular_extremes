import sys
import numpy as np

from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="llama3")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma

persist_directory = "./chroma_db"
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

def search(i):
  top = np.zeros(4096)
  bottom = np.zeros(4096)
  top[i] = 1
  bottom[i] = -1
  top = vectorstore.similarity_search_by_vector(top)
  bottom = vectorstore.similarity_search_by_vector(bottom)
  print("%04d" % (i), -1, bottom[0].page_content, top[0].page_content, 1)

zero_vector = np.zeros(4096)
ones_vector = np.ones(4096)
negative_ones_vector = np.ones(4096) * -1

zero = vectorstore.similarity_search_by_vector(zero_vector)
ones = vectorstore.similarity_search_by_vector(ones_vector)
negative_ones = vectorstore.similarity_search_by_vector(negative_ones_vector)

print("0s", zero[0].page_content)
print("1s", ones[0].page_content)
print("-1s", negative_ones[0].page_content)
print()

# Get raw vectors from the vectorstore
docs = vectorstore.get(include=['embeddings', 'documents'])
if docs and len(docs['embeddings']) > 0:
    first_word = docs['documents'][0]  # This is the raw numpy array
    first_vector = docs['embeddings'][0]  # This is the raw numpy array
    print("First word:", first_word)
    print("First vector shape:", first_vector.shape)
    print("First few values:", first_vector[:10])
print()

print('total docs:', len(docs['documents']))

for i in range(4096):
  min_val = 0
  max_val = 0
  min_word = None
  max_word = None
  for j in range(len(docs['documents'])):
    word = docs['documents'][j]  # This is the raw numpy array
    vector = docs['embeddings'][j]  # This is the raw numpy array
    if vector[i] < min_val:
      min_val = vector[i]
      min_word = word
    if vector[i] > max_val:
      max_val = vector[i]
      max_word = word
  print(i, min_val, min_word, max_word, max_val)

print(docs['embeddings'][0])