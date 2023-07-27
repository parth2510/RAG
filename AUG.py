import requests
import re
import openai
import pandas as pd
import time
import fitz
import tiktoken
import ast

from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

#new Knowledge document to make embeddings
def change_doc(document_path, title, key):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    doc = fitz.open(document_path)
    text = ""
    for page in doc:
        text+=page.get_text()

    def preprocess_text(t):
        
        t = t.replace('\n', ' ')
        t = t.replace('\n', ' ')
        t = t.replace('  ', ' ')
        t = t.replace('  ', ' ')

        #remove images?
        return t

    processed_text = preprocess_text(text)

    chunks = chunk_it(processed_text, title) #make chunks

    df = pd.DataFrame(chunks, columns = ['text'])
    df['n_tokens'] = df.text.map(lambda x: len(tokenizer.encode(x)))
    # df.n_tokens.hist()

    # Create embeddings for each text in the DataFrame with throttling
    df['embeddings'] = df['text'].apply(lambda x :throttled_api_call(x,key))
    df.to_csv(f'{title}_embeddings.csv')

# function to make chunks of our doc
def chunk_it(text, title):
    final_chunks = []
    max_tokens = 500
    delimeter = "###" # delimeter which fits well for seperating questions (except faq) sections for given document
    

    sections = text.split(delimeter) 

    #if the text has noise seperated by ###, well use a different chunking strategy

    n_sections = len(sections) #since no.of sections is less, we can try to chunk it sectionwise for these kinds & size of documents
    
    # Get the number of tokens for each sections
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_tokens = [len(tokenizer.encode(" " + section)) for section in sections]

    for section, token in zip(sections, n_tokens): 

        if (token <3):
            continue #noise or mistakes, make sure there are not many

        if token < max_tokens:
            final_chunks.append("Title: " + str(title) +"." + section) # appending title in every chunk for global context

        # # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if token > max_tokens:
            final_chunks += split_into_many(section, title, max_tokens)
    
    return final_chunks

# Function to split the text section into multiple chunks of a maximum number of tokens
def split_into_many(text, title, max_tokens = 500):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):

        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence # unlikely
        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

# Throttle function to control the rate of API calls
def throttled_api_call(text, key):
    openai.api_key = key
    print("creating embeddings of the doc")
    response = openai.Embedding.create(input=text, engine='text-embedding-ada-002')
    time.sleep(21)  # Adjust the sleep duration to stay within your rate limit,  mine is 3 per min
    print("done")
    return response['data'][0]['embedding']


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        if cur_len > max_len:
            break
        
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    question="Am I a robot?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # prompt = f"""Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"
    prompt=f"Answer the question ellaborately and truthfully. If the information about the question in not present in the context, explain that you are  unable to answer it as it is out of context. Context: {context}\n Question: {question}"

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        temperature = 0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""


def ask(question,key, title = 'PAN', new_knowledge_doc = False):
    if(new_knowledge_doc == True):
        # add a new knowledge document
        change_doc("bert5.pdf", title, key)


    openai.api_key = key
    try:
        df = pd.read_csv(f'{title}_embeddings.csv')
        df['embeddings'] = df['embeddings'].apply(ast.literal_eval)

        return answer_question(df, question, debug=True)
    except Exception as e:
        print("knowledge document not found, please see instructions to add a new knowledge document ")
        print(e)
        return "error, see logs" + str(e)

    

# print("ans", ask("What is BERT ", "api_key", "BERT"))
    