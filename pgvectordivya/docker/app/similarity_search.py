import pandas as pd
from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client
import logging

logging.basicConfig(
    level=logging.WARNING,  # Change from INFO to WARNING
    format="%(message)s"    # Keep the log format simple
)

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Relevant question from the dataset
# --------------------------------------------------------------

relevant_question = "What is the Governing Law of VIDEO-ON-DEMAND CONTENT LICENSE AGREEMENT?"
results = vec.search(relevant_question, limit=3)

# No need to transform results - pass DataFrame directly
response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response['answer']}")
print("\nThought process:")
for thought in response['thought_process']:
    print(f"- {thought}")
print(f"\nContext: {response['enough_context']}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "What is the weather in Tokyo?"

results = vec.search(irrelevant_question, limit=3)

# Pass results to the Synthesizer
response = Synthesizer.generate_response(question=irrelevant_question, context=results)

# Print the response
print(f"\n{response['answer']}")
print("\nThought process:")
for thought in response['thought_process']:
    print(f"- {thought}")
print(f"\nContext: {response['enough_context']}")


# --------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------

metadata_filter = {"Document Name": "Master Agreement"}

results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

# Pass filtered results to Synthesizer
response = Synthesizer.generate_response(question=relevant_question, context=results)


# Print the response
print(f"\n{response['answer']}")
print("\nThought process:")
for thought in response['thought_process']:
    print(f"- {thought}")
if response['enough_context']:
    print(f"\n{response['answer']}")
else:
    print("No relevant context found.")    



# --------------------------------------------------------------
# Advanced filtering using Predicates
# --------------------------------------------------------------

predicates = client.Predicates("Document Name", "==", "Master Agreement")
results = vec.search(relevant_question, limit=3, predicates=predicates)

# Advanced predicate filtering
predicates = client.Predicates("Document Name", "==", "Master Agreement") | client.Predicates(
    "Parties", "==", "Party A and Party B"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)

# Advanced filtering with time-based predicates
predicates = client.Predicates("Document Name", "==", "Master Agreement") & client.Predicates(
    "created_at", ">", "2024-09-01"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)


# --------------------------------------------------------------
# Time-based filtering
# --------------------------------------------------------------

# September — Returning results
time_range = (datetime(2024, 9, 1), datetime(2024, 9, 30))
results = vec.search(relevant_question, limit=3, time_range=time_range)

# August — Not returning any results
time_range = (datetime(2024, 8, 1), datetime(2024, 8, 30))
results = vec.search(relevant_question, limit=3, time_range=time_range)
