import numpy as np

from langchain_ollama import OllamaEmbeddings
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

class MaxThree:
  def __init__(self):
      self.values = []

  def add(self, value, item):
    self.values.append([value, item])
    self.values.sort(key=lambda x: x[0])
    if len(self.values) > 3:
      self.values.pop(0)
  
  def __str__(self):
    return f"{self.values[0][1]},{self.values[1][1]},{self.values[2][1]} {self.values[2][0]}"

class MinThree:
  def __init__(self):
      self.values = []

  def add(self, value, item):
    self.values.append([value, item])
    self.values.sort(key=lambda x: x[0])
    if len(self.values) > 3:
      self.values.pop()
  
  def __str__(self):
    return f"{self.values[0][0]} {self.values[0][1]},{self.values[1][1]},{self.values[2][1]}"

for i in range(4096):
  min_three = MinThree()
  max_three = MaxThree()
  for j in range(len(docs['documents'])):
    word = docs['documents'][j]  # This is the raw numpy array
    vector = docs['embeddings'][j]  # This is the raw numpy array
    vector_val = vector[i]
    min_three.add(vector_val, word)
    max_three.add(vector_val, word)
  print(i, str(min_three), str(max_three))

print(docs['embeddings'][0])