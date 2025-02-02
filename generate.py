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

with open("./adjectives.txt") as words_file:
  words = words_file.readlines()

words = [word.strip().lower().split('\t')[0] for word in words]
print(words)
words = [Document(page_content=word, id=word) for word in words]

start = 0
batch_size = 100
total = len(words)
remaining = total

while remaining > 0:
    end = min(start + batch_size, total)
    batch = words[start:end]
    ids = [str(uuid4()) for _ in range(len(batch))]
    vectorstore.add_documents(documents=batch, ids=ids)
    
    added = end - start
    remaining -= added
    print(f"Added {added} words. {remaining} remaining out of {total}")
    start = end

zero_vector = np.zeros(4096)
ones_vector = np.ones(4096)
negative_ones_vector = np.ones(4096) * -1

def search(tag, vector):
  print(tag)
  results = vectorstore.similarity_search_by_vector(vector)
  for doc in results:
      print(doc)
  print()

search("0s", zero_vector)
search("1s", ones_vector)
search("-1s", negative_ones_vector)