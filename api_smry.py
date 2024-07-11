# Loads the case/reply embeddings
# Embeds the query
# Finds the matches
# Generates a summary of all three cases
# Generate a summary of each case
# Return the summary with the case ID's

from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import numpy as np
import pandas as pd
import ast

def get_knowledge_base_embeddings():
    data = pd.read_csv("/Users/heer/Projects/Semantic_Search/scripts/data.csv")
    data['embeddings'] = data['embeddings'].apply(ast.literal_eval)
    column_series = data['embeddings']
    embeddings = column_series.tolist()
    return embeddings, data

def get_embedding(sentence):
    embedding = client.embeddings.create(
    model="text-embedding-ada-002",
    input=sentence
    )
    return embedding.data[0].embedding

def find_closest_match(query, embeddings, data):#, sentences):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings)
    # closest_index = np.argmax(similarities)
    top_indices = np.argsort(similarities)
    top3_indices = top_indices[0][-3:]
    one = top3_indices[0]
    two = top3_indices[1]
    three = top3_indices[2]

    top_case_1 = data.iloc[one]
    top_case_2 = data.iloc[two]
    top_case_3 = data.iloc[three]

    return top_case_1, top_case_2, top_case_3 

def generate_conversational_reply(user_query, case_text, reply_text):
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are an doctor who gives summaries about medical cases. You only provide answers under the following headings: Summary, Patient, Symptoms, Medical History, diagnosis, treatment"},
            {"role": "user", "content": f"First provide 100 words summarising this data: {case_text} in in reply to this question: {user_query}. Secondly extract bullet points about the Patient, Symptoms and Medical History from this data: {case_text}. Finally extract bullet points about the diagnosis, treatment options from this data: {reply_text}"}
        ]
    )
    return chat_completion.choices[0].message.content

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query', '')

    embeddings, data = get_knowledge_base_embeddings()
    top_case_1, top_case_2, top_case_3 = find_closest_match(user_query, embeddings=embeddings, data=data)#, sentences=data['case_text'] + data['reply_text'])

    result1 = generate_conversational_reply(user_query, top_case_1['case_text'], top_case_1['reply_text'])
    result2 = generate_conversational_reply(user_query, top_case_2['case_text'], top_case_2['reply_text'])
    result3 = generate_conversational_reply(user_query, top_case_3['case_text'], top_case_3['reply_text'])
    result4 = generate_conversational_reply(user_query, top_case_1['case_text'] + top_case_2['case_text'] + top_case_3['case_text'], top_case_1['reply_text'] + top_case_2['reply_text'] + top_case_3['reply_text'])


    return jsonify({
        'case_id_1': str(top_case_1['case_id']),
        'case_id_2': str(top_case_2['case_id']),
        'case_id_3': str(top_case_3['case_id']),

        'summary_1':result1,
        'summary_2':result2,
        'summary_3':result3,

        'agg_summary':result4
        })

if __name__ == '__main__':
    app.run(debug=True)
