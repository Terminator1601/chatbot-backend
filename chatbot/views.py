
# from django.shortcuts import render
# from django.conf import settings
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import re
# import os
# from spellchecker import SpellChecker
# import json
# import traceback
# import datetime
# import psycopg2
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# from django.core.files.storage import default_storage
# from django.views.decorators.http import require_http_methods
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import re
# import psycopg2
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from spellchecker import SpellChecker

# # Define your dictionary with common words, network troubleshooting terms, and spellcheck
# network_troubleshooting_terms = {
#     'ping', 'traceroute', 'firewall', 'router', 'switch', 'network', 'IP', 'subnet', 'DNS', 'DHCP', 'bandwidth',
#     'latency', 'throughput', 'packet loss', 'VPN', 'TCP', 'UDP', 'port', 'protocol', 'LAN', 'WAN', 'NAT', 'SNAS', 'BARC', 'ipv4', 'ipv6',
# }

# # Initialize SpellChecker and add network troubleshooting terms
# custom_dict = SpellChecker()
# custom_dict.word_frequency.load_words(network_troubleshooting_terms)

# # Initialize the TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = None
# df = None

# # Function to load data from PostgreSQL database
# def load_data_from_db():
#     global df, tfidf_matrix

#     try:
#         # Establish the database connection
#         conn = psycopg2.connect(
#             dbname='chatbot',
#             user='postgres'
#         )
#         cur = conn.cursor()

#         # Fetch the data from the database
#         cur.execute("""
#             SELECT question, answer, media_name
#             FROM chatbot_data
#         """)
#         rows = cur.fetchall()
#         df = pd.DataFrame(rows, columns=['question', 'answer', 'media_name'])
#         df['question'] = df['question'].str.lower()
#         df['answer'] = df['answer'].str.lower()

#         # Initialize the TF-IDF vectorizer
#         global vectorizer
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform(df['question'])

#         cur.close()
#         conn.close()

#     except Exception as e:
#         print(f"Error occurred while loading data from the database: {e}")
#         raise  # Ensure any exception causes the process to fail and get noticed

# # Call the load_data_from_db function when the module is imported
# load_data_from_db()

# # Function to detect meaningless sentences
# def is_meaningless(sentence):
#     if len(sentence) < 2:  # too short to be meaningful
#         return True
#     words = set(re.findall(r'\b\w+\b', sentence))
#     # Avoid flagging common phrases or greetings as meaningless
#     common_phrases = {'hi', 'hello', 'how', 'are', 'you', 'good', 'morning', 'evening', 'bye', 'thanks', 'thank'}
#     if words & common_phrases:
#         return False
#     if words & network_troubleshooting_terms:  # contains network troubleshooting terms
#         return False
#     return True

# # Define the vector match threshold
# VECTOR_MATCH_THRESHOLD = 0.8

# # Function to find the answer
# def find_answer(query):
#     questions = re.split(
#         r'\s*(?:[?.!,;]\s*| and | or | if | when | but | unless | because | while | since | so | although | whether )\s*',
#         query.strip(), flags=re.IGNORECASE
#     )

#     suggestions = []
#     answers = []
#     media = None  # Initialize media to None

#     for question in questions:
#         if question:
#             question_tfidf = vectorizer.transform([question.lower()])
#             cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
#             best_match_idx = cosine_similarities.argmax()
#             if cosine_similarities[best_match_idx] > VECTOR_MATCH_THRESHOLD:
#                 answers.append(df['answer'].iloc[best_match_idx])
#                 media = df['media_name'].iloc[best_match_idx]  # Get the media name of the best match
#             else:
#                 answer = fallback_answer(question)
#                 # No need to add fallback answers here
#                 # suggestions.append(answer)

#     # Get suggestions for invalid words
#     invalid_words = re.findall(r'\b\w+\b', query)
#     invalid_words = [word for word in invalid_words if not custom_dict.known([word])]
#     for word in invalid_words:
#         close_matches = custom_dict.candidates(word)
#         if close_matches:
#             suggestions.extend(close_matches)  # Directly add the list of matches
#         else:
#             suggestions.append(f"No suggestions for '{word}'.")

#     return answers, suggestions, media, find_best_matches(query)  # Add media and best matches

# # Function to find the best three matching questions from the dataset
# def find_best_matches(query):
#     query_tfidf = vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
#     best_match_indices = cosine_similarities.argsort()[-6:-1]  # Get the top 5 matches excluding the highest match

#     best_matches = df['question'].iloc[best_match_indices].tolist()
#     return best_matches

# # Fallback function for out-of-dataset questions
# def fallback_answer(query):
#     query_tfidf = vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
#     best_match_idx = cosine_similarities.argsort()[-2]
#     return df['answer'].iloc[best_match_idx]

# @csrf_exempt
# def chatbot_view(request):
#     if request.method == 'POST':
#         user_query = request.POST.get('query')
#         if not user_query:
#             return JsonResponse({'error': 'No query provided'}, status=400)

#         # Check if tfidf_matrix and vectorizer are initialized
#         if tfidf_matrix is None or vectorizer is None:
#             load_data_from_db()  # Ensure that data is loaded and TF-IDF vectorizer is fitted

#         try:
#             words = re.findall(r'\b\w+\b', user_query)
#             valid_words = [word for word in words if custom_dict.known([word])]
#             answers, suggestions, media, best_matches = find_answer(user_query)

#             response_data = {}

#             if not valid_words:
#                 # Return only suggestions for invalid queries
#                 response_data['suggestions'] = suggestions  # Directly use the list of suggestions
#             else:
#                 # Return answers if valid and available
#                 response_data['answers'] = answers

#             # Add suggestions to the response data
#             if suggestions:
#                 response_data['suggestions'] = suggestions

#             # Add best matches to the response data if no valid answers are found
#             if not answers:
#                 response_data['best_matches'] = best_matches

#             # Add media_name to the response data if it exists and there are valid answers
#             if media and answers:
#                 response_data['media_name'] = media

#             return JsonResponse(response_data)
#         except Exception as e:
#             # Log the exception and return a 500 error response
#             print(f"Error occurred: {e}")
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)

#     return render(request, 'chatbot/chatbot.html')


# @csrf_exempt
# def feedback_view(request):
#     if request.method == 'POST':
#         try:
#             feedback_data = json.loads(request.body)
#             feedback = feedback_data.get('feedback')
#             previous_input = feedback_data.get('previousInput')

#             if not feedback or not previous_input:
#                 return JsonResponse({'error': 'Feedback or previous input is missing'}, status=400)

#             print(f"Request data: {feedback_data}")

#             # Establish connection to the PostgreSQL database
#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()

#             # Retrieve the generated id
#             cur.execute("SELECT MAX(id) FROM feedback_logs")
#             nId = cur.fetchone()[0]
#             if nId is None:
#                 new_id = 1
#             else:
#                 new_id = nId + 1

#             print(f"Inserting question: {previous_input}, answer: {feedback} in id: {new_id}")

#             # Insert the new question, answer, and timestamp into the "feedback_logs" table
#             insert_query = """
#             INSERT INTO feedback_logs (timestamp, feedback, previous_input, id)
#             VALUES (%s, %s, %s, %s)
#             RETURNING id
#             """
#             cur.execute(insert_query, (datetime.now(), feedback, previous_input, new_id))

#             # Commit the transaction
#             conn.commit()

#             print("Data inserted successfully with ID:", new_id)

#             cur.close()
#             conn.close()

#             return JsonResponse({'status': 'success', 'id': new_id})
#         except psycopg2.Error as db_err:
#             print(f"Database error occurred: {db_err}")
#             print(traceback.format_exc())
#             return JsonResponse({'error': 'Database Error'}, status=500)
#         except Exception as e:
#             print(f"Error occurred while processing feedback: {e}")
#             print(traceback.format_exc())
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)


# @csrf_exempt
# def get_feedback_logs(request):
#     if request.method == 'GET':
#         try:
#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()
#             cur.execute("SELECT id, timestamp, feedback, previous_input FROM feedback_logs")
#             rows = cur.fetchall()
#             logs = [
#                 {
#                     'id': row[0],
#                     'timestamp': row[1].isoformat(),  # Ensure timestamp is in ISO format
#                     'feedback': row[2],
#                     'previous_input': row[3]
#                 }
#                 for row in rows
#             ]
#             cur.close()
#             conn.close()
#             return JsonResponse(logs, safe=False)
#         except Exception as e:
#             print(f"Error fetching feedback logs: {e}")
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)
#     return JsonResponse({'error': 'Invalid request method'}, status=400)


# @csrf_exempt
# def delete_log(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             feedback = data.get('feedback')
#             previous_input = data.get('previousInput')
#             source = data.get('source')
#             media=data.get('media')
#             Id=data.get('Id')
#             # print(feedback)
#             # print(previous_input)

#             if not feedback or not previous_input:
#                 return JsonResponse({'error': 'Feedback or previous input is missing'}, status=400)
#             # Connect to the PostgreSQL database
#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()

#             print(f"Inserting question: {previous_input}, answer: {feedback}, source{source}, media: {media}")

#             # Insert the new question and answer into the "chatbot_data" table
#             insert_query = """
#             INSERT INTO chatbot_data (question, answer, media_name, source)
#             VALUES (%s, %s, %s, %s)
#             """

#             cur.execute(insert_query, (previous_input, feedback, media, source))


#             # Remove the log from the "feedback_logs" table
#             delete_query = f"""
#             DELETE FROM feedback_logs
#             WHERE id = '{Id}'
#             AND feedback = 'not helpful'
#             """
#             print(delete_query)
#             data = cur.execute(delete_query)
#             conn.commit()
#             cur.close()
#             conn.close()
#             return JsonResponse({'status': 'success'})

#         except Exception as e:
#             print(f"Error occurred while processing feedback: {e}")
#             print(traceback.format_exc())
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)
#     load_data_from_db()


# @csrf_exempt
# def reject_log(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             log_id = data.get('id')
#             feedback = data.get('feedback', 'not helpful')  # Default feedback if not provided

#             if not log_id:
#                 return JsonResponse({'error': 'Missing log ID'}, status=400)

#             # Connect to the PostgreSQL database
#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()

#             # Remove the log from the "feedback_logs" table
#             delete_query = """
#             DELETE FROM feedback_logs
#             WHERE id = %s AND feedback = %s
#             """
#             cur.execute(delete_query, (log_id, feedback))
#             conn.commit()
#             cur.close()
#             conn.close()
#             return JsonResponse({'status': 'success'})

#         except Exception as e:
#             print(f"Error occurred while processing feedback: {e}")
#             print(traceback.format_exc())
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)
#     load_data_from_db()


# @csrf_exempt
# def add_new_data(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             newQuestion = data.get('question')
#             newAnswer = data.get('answer')
#             newMedia = data.get('media')
#             newSource=data.get('source')
#             print(newQuestion)
#             print(newAnswer)
#             print(newMedia)
#             print(newSource)


#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()

#             print(f"Inserting question: {newQuestion}, answer: {newAnswer}, media: {newMedia}, source: {newSource}")

#             # Insert the new question and answer into the "chatbot_data" table
#             insert_query = """
#             INSERT INTO chatbot_data (question, answer, media_name, source)
#             VALUES (%s, %s, %s, %s)
#             """

#             cur.execute(insert_query, (newQuestion, newAnswer, newMedia, newSource))

#             conn.commit()
#             cur.close()
#             conn.close()
#             return JsonResponse({'status': 'success'})

#         except Exception as e:
#             print(f"Error occurred while processing feedback: {e}")
#             print(traceback.format_exc())
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)
#     load_data_from_db()


# @csrf_exempt
# def add_media(request):
#     if request.method == 'POST':
#         file = request.FILES.get('file')
#         log_id = request.POST.get('logId')

#         if file and log_id:
#             # Define the path where you want to save the file
#             file_path = f'images/{log_id}_{file.name}'
#             with default_storage.open(file_path, 'wb+') as destination:
#                 for chunk in file.chunks():
#                     destination.write(chunk)

#             # Return a JSON response
#             return JsonResponse({'message': 'File uploaded successfully', 'file_path': file_path})
#         else:
#             return JsonResponse({'error': 'No file or logId provided'}, status=400)
#     else:
#         return JsonResponse({'error': 'Invalid request method'}, status=405)


# @csrf_exempt
# def add_new_media(request):
#     if request.method == 'POST':
#         try:
#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()

#             # Retrieve the generated id
#             cur.execute("SELECT MAX(id) FROM chatbot_data")
#             nId = cur.fetchone()[0]
#             if nId is None:
#                 new_id = 1
#             else:
#                 new_id = nId + 1

#             file = request.FILES.get('file')

#             if file and new_id:
#                 # Define the path where you want to save the file
#                 file_path = f'images/{new_id}_{file.name}'
#                 with default_storage.open(file_path, 'wb+') as destination:
#                     for chunk in file.chunks():
#                         destination.write(chunk)

#                 # Return a JSON response
#                 return JsonResponse({'message': 'File uploaded successfully', 'file_path': file_path})
#             else:
#                 return JsonResponse({'error': 'No file or new_id provided'}, status=400)
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#     else:
#         return JsonResponse({'error': 'Invalid request method'}, status=405)


# @csrf_exempt
# def get_table_data(request):
#     try:
#         # Connect to your PostgreSQL database
#         conn = psycopg2.connect(
#             dbname="chatbot",
#             user="postgres"
#         )

#         # Use context manager to ensure cursor and connection are properly closed
#         with conn.cursor() as cursor:
#             # Execute your query
#             cursor.execute("SELECT id, question, answer, media_name, source FROM chatbot_data")
#             rows = cursor.fetchall()

#             # Get column names
#             colnames = [desc[0] for desc in cursor.description]

#             # Create a list of dictionaries with column names as keys
#             data = [dict(zip(colnames, row)) for row in rows]

#         # Close connection explicitly (context manager takes care of cursor)
#         conn.close()

#         return JsonResponse(data, safe=False)  # safe=False allows non-dict objects to be serialized
#     except psycopg2.Error as e:
#         # Handle database-related errors
#         return JsonResponse({'error': f'Database error: {str(e)}'}, status=500)
#     except Exception as e:
#         # Handle other unexpected errors
#         return JsonResponse({'error': str(e)}, status=500)


# @csrf_exempt
# def delete_data(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             log_id = data.get('id')
#             # feedback = data.get('feedback', 'not helpful')  # Default feedback if not provided

#             if not log_id:
#                 return JsonResponse({'error': 'Missing log ID'}, status=400)

#             print('id ' ,id)
#             # Connect to the PostgreSQL database
#             conn = psycopg2.connect(
#                 dbname='chatbot',
#                 user='postgres'
#             )
#             cur = conn.cursor()

#             # Remove the log from the "feedback_logs" table
#             delete_query = """
#             DELETE FROM chatbot_data
#             WHERE id = %s ;
#             """
#             cur.execute(delete_query, (log_id,))
#             conn.commit()
#             cur.close()
#             conn.close()
#             return JsonResponse({'status': 'success'})

#         except Exception as e:
#             print(f"Error occurred while processing feedback: {e}")
#             print(traceback.format_exc())
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=400)
#     load_data_from_db()


from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
import os
from spellchecker import SpellChecker
import json
import traceback
import datetime
import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from django.core.files.storage import default_storage
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

# Define your dictionary with common words, network troubleshooting terms, and spellcheck
network_troubleshooting_terms = {
    'ping', 'traceroute', 'firewall', 'router', 'switch', 'network', 'IP', 'subnet', 'DNS', 'DHCP', 'bandwidth',
    'latency', 'throughput', 'packet loss', 'VPN', 'TCP', 'UDP', 'port', 'protocol', 'LAN', 'WAN', 'NAT', 'SNAS', 'BARC', 'ipv4', 'ipv6',
}

# Initialize SpellChecker and add network troubleshooting terms
custom_dict = SpellChecker()
custom_dict.word_frequency.load_words(network_troubleshooting_terms)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = None
df = None

# Function to load data from PostgreSQL database


def load_data_from_db():
    global df, tfidf_matrix

    try:
        # Establish the database connection
        conn = psycopg2.connect(
            dbname='chatbot',
            user='postgres',
            password='admin',
            port='5432'
        )
        cur = conn.cursor()

        # Fetch the data from the database
        cur.execute("""
            SELECT question, answer, media_name
            FROM chatbot_data
        """)
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=['question', 'answer', 'media_name'])
        df['question'] = df['question'].str.lower()
        df['answer'] = df['answer'].str.lower()

        # Initialize the TF-IDF vectorizer
        global vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['question'])

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error occurred while loading data from the database: {e}")
        raise  # Ensure any exception causes the process to fail and get noticed


# Call the load_data_from_db function when the module is imported
load_data_from_db()

# Function to detect meaningless sentences


def is_meaningless(sentence):
    if len(sentence) < 2:  # too short to be meaningful
        return True
    words = set(re.findall(r'\b\w+\b', sentence))
    # Avoid flagging common phrases or greetings as meaningless
    common_phrases = {'hi', 'hello', 'how', 'are', 'you',
                      'good', 'morning', 'evening', 'bye', 'thanks', 'thank'}
    if words & common_phrases:
        return False
    if words & network_troubleshooting_terms:  # contains network troubleshooting terms
        return False
    return True


# Define the vector match threshold
VECTOR_MATCH_THRESHOLD = 0.8

# Function to find the answer


def find_answer(query):
    questions = re.split(
        r'\s*(?:[?.!,;]\s*| and | or | if | when | but | unless | because | while | since | so | although | whether )\s*',
        query.strip(), flags=re.IGNORECASE
    )

    suggestions = []
    answers = []
    media = None  # Initialize media to None

    for question in questions:
        if question:
            question_tfidf = vectorizer.transform([question.lower()])
            cosine_similarities = cosine_similarity(
                question_tfidf, tfidf_matrix).flatten()
            best_match_idx = cosine_similarities.argmax()
            if cosine_similarities[best_match_idx] > VECTOR_MATCH_THRESHOLD:
                answers.append(df['answer'].iloc[best_match_idx])
                # Get the media name of the best match
                media = df['media_name'].iloc[best_match_idx]
            else:
                answer = fallback_answer(question)
                # No need to add fallback answers here
                # suggestions.append(answer)

    # Get suggestions for invalid words
    invalid_words = re.findall(r'\b\w+\b', query)
    invalid_words = [word for word in invalid_words if not custom_dict.known([
                                                                             word])]
    for word in invalid_words:
        close_matches = custom_dict.candidates(word)
        if close_matches:
            # Directly add the list of matches
            suggestions.extend(close_matches)
        else:
            suggestions.append(f"No suggestions for '{word}'.")

    # Add media and best matches
    return answers, suggestions, media, find_best_matches(query)

# Function to find the best three matching questions from the dataset


def find_best_matches(query):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(
        query_tfidf, tfidf_matrix).flatten()
    # Get the top 5 matches excluding the highest match
    best_match_indices = cosine_similarities.argsort()[-6:-1]

    best_matches = df['question'].iloc[best_match_indices].tolist()
    return best_matches

# Fallback function for out-of-dataset questions


def fallback_answer(query):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(
        query_tfidf, tfidf_matrix).flatten()
    best_match_idx = cosine_similarities.argsort()[-2]
    return df['answer'].iloc[best_match_idx]


@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        user_query = request.POST.get('query')
        if not user_query:
            return JsonResponse({'error': 'No query provided'}, status=400)

        # Check if tfidf_matrix and vectorizer are initialized
        if tfidf_matrix is None or vectorizer is None:
            load_data_from_db()  # Ensure that data is loaded and TF-IDF vectorizer is fitted

        try:
            words = re.findall(r'\b\w+\b', user_query)
            valid_words = [word for word in words if custom_dict.known([word])]
            answers, suggestions, media, best_matches = find_answer(user_query)

            response_data = {}

            if not valid_words:
                # Return only suggestions for invalid queries
                # Directly use the list of suggestions
                response_data['suggestions'] = suggestions
            else:
                # Return answers if valid and available
                response_data['answers'] = answers

            # Add suggestions to the response data
            if suggestions:
                response_data['suggestions'] = suggestions

            # Add best matches to the response data if no valid answers are found
            if not answers:
                response_data['best_matches'] = best_matches

            # Add media_name to the response data if it exists and there are valid answers
            if media and answers:
                response_data['media_name'] = media

            return JsonResponse(response_data)
        except Exception as e:
            # Log the exception and return a 500 error response
            print(f"Error occurred: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

    return render(request, 'chatbot/chatbot.html')


@csrf_exempt
def feedback_view(request):
    if request.method == 'POST':
        try:
            feedback_data = json.loads(request.body)
            feedback = feedback_data.get('feedback')
            previous_input = feedback_data.get('previousInput')

            if not feedback or not previous_input:
                return JsonResponse({'error': 'Feedback or previous input is missing'}, status=400)

            print(f"Request data: {feedback_data}")

            # Establish connection to the PostgreSQL database
            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()

            # Retrieve the generated id
            cur.execute("SELECT MAX(id) FROM feedback_logs")
            nId = cur.fetchone()[0]
            if nId is None:
                new_id = 1
            else:
                new_id = nId + 1

            print(f"Inserting question: {previous_input}, answer: {
                  feedback} in id: {new_id}")

            # Insert the new question, answer, and timestamp into the "feedback_logs" table
            insert_query = """
            INSERT INTO feedback_logs (timestamp, feedback, previous_input, id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(insert_query, (datetime.now(),
                        feedback, previous_input, new_id))

            # Commit the transaction
            conn.commit()

            print("Data inserted successfully with ID:", new_id)

            cur.close()
            conn.close()

            return JsonResponse({'status': 'success', 'id': new_id})
        except psycopg2.Error as db_err:
            print(f"Database error occurred: {db_err}")
            print(traceback.format_exc())
            return JsonResponse({'error': 'Database Error'}, status=500)
        except Exception as e:
            print(f"Error occurred while processing feedback: {e}")
            print(traceback.format_exc())
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def get_feedback_logs(request):
    if request.method == 'GET':
        try:
            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT id, timestamp, feedback, previous_input FROM feedback_logs")
            rows = cur.fetchall()
            logs = [
                {
                    'id': row[0],
                    # Ensure timestamp is in ISO format
                    'timestamp': row[1].isoformat(),
                    'feedback': row[2],
                    'previous_input': row[3]
                }
                for row in rows
            ]
            cur.close()
            conn.close()
            return JsonResponse(logs, safe=False)
        except Exception as e:
            print(f"Error fetching feedback logs: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def delete_log(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            feedback = data.get('feedback')
            previous_input = data.get('previousInput')
            source = data.get('source')
            media = data.get('media')
            Id = data.get('Id')
            # print(feedback)
            # print(previous_input)

            if not feedback or not previous_input:
                return JsonResponse({'error': 'Feedback or previous input is missing'}, status=400)
            # Connect to the PostgreSQL database
            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()

            print(f"Inserting question: {previous_input}, answer: {
                  feedback}, source{source}, media: {media}")

            # Insert the new question and answer into the "chatbot_data" table
            insert_query = """
            INSERT INTO chatbot_data (question, answer, media_name, source)
            VALUES (%s, %s, %s, %s)
            """

            cur.execute(insert_query, (previous_input,
                        feedback, media, source))

            # Remove the log from the "feedback_logs" table
            delete_query = f"""
            DELETE FROM feedback_logs
            WHERE id = '{Id}'
            AND feedback = 'not helpful'
            """
            print(delete_query)
            data = cur.execute(delete_query)
            conn.commit()
            cur.close()
            conn.close()
            return JsonResponse({'status': 'success'})

        except Exception as e:
            print(f"Error occurred while processing feedback: {e}")
            print(traceback.format_exc())
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
    load_data_from_db()


@csrf_exempt
def reject_log(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            log_id = data.get('id')
            # Default feedback if not provided
            feedback = data.get('feedback', 'not helpful')

            if not log_id:
                return JsonResponse({'error': 'Missing log ID'}, status=400)

            # Connect to the PostgreSQL database
            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()

            # Remove the log from the "feedback_logs" table
            delete_query = """
            DELETE FROM feedback_logs
            WHERE id = %s AND feedback = %s
            """
            cur.execute(delete_query, (log_id, feedback))
            conn.commit()
            cur.close()
            conn.close()
            return JsonResponse({'status': 'success'})

        except Exception as e:
            print(f"Error occurred while processing feedback: {e}")
            print(traceback.format_exc())
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
    load_data_from_db()


@csrf_exempt
def add_new_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            newQuestion = data.get('question')
            newAnswer = data.get('answer')
            newMedia = data.get('media')
            newSource = data.get('source')
            print(newQuestion)
            print(newAnswer)
            print(newMedia)
            print(newSource)

            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()

            print(f"Inserting question: {newQuestion}, answer: {
                  newAnswer}, media: {newMedia}, source: {newSource}")

            # Insert the new question and answer into the "chatbot_data" table
            insert_query = """
            INSERT INTO chatbot_data (question, answer, media_name, source)
            VALUES (%s, %s, %s, %s)
            """

            cur.execute(insert_query, (newQuestion,
                        newAnswer, newMedia, newSource))

            conn.commit()
            cur.close()
            conn.close()
            return JsonResponse({'status': 'success'})

        except Exception as e:
            print(f"Error occurred while processing feedback: {e}")
            print(traceback.format_exc())
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
    load_data_from_db()


@csrf_exempt
def add_media(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        log_id = request.POST.get('logId')

        if file and log_id:
            # Define the path where you want to save the file
            file_path = f'images/{log_id}_{file.name}'
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # Return a JSON response
            return JsonResponse({'message': 'File uploaded successfully', 'file_path': file_path})
        else:
            return JsonResponse({'error': 'No file or logId provided'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def add_new_media(request):
    if request.method == 'POST':
        try:
            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()

            # Retrieve the generated id
            cur.execute("SELECT MAX(id) FROM chatbot_data")
            nId = cur.fetchone()[0]
            if nId is None:
                new_id = 1
            else:
                new_id = nId + 1

            file = request.FILES.get('file')

            if file and new_id:
                # Define the path where you want to save the file
                file_path = f'images/{new_id}_{file.name}'
                with default_storage.open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

                # Return a JSON response
                return JsonResponse({'message': 'File uploaded successfully', 'file_path': file_path})
            else:
                return JsonResponse({'error': 'No file or new_id provided'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def get_table_data(request):
    try:
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(
            dbname="chatbot",
            user="postgres",
            password='admin',
            port='5432',
        )

        # Use context manager to ensure cursor and connection are properly closed
        with conn.cursor() as cursor:
            # Execute your query
            cursor.execute(
                "SELECT id, question, answer, media_name, source FROM chatbot_data")
            rows = cursor.fetchall()

            # Get column names
            colnames = [desc[0] for desc in cursor.description]

            # Create a list of dictionaries with column names as keys
            data = [dict(zip(colnames, row)) for row in rows]

        # Close connection explicitly (context manager takes care of cursor)
        conn.close()

        # safe=False allows non-dict objects to be serialized
        return JsonResponse(data, safe=False)
    except psycopg2.Error as e:
        # Handle database-related errors
        return JsonResponse({'error': f'Database error: {str(e)}'}, status=500)
    except Exception as e:
        # Handle other unexpected errors
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def delete_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            log_id = data.get('id')
            # feedback = data.get('feedback', 'not helpful')  # Default feedback if not provided

            if not log_id:
                return JsonResponse({'error': 'Missing log ID'}, status=400)

            print('id ', id)
            # Connect to the PostgreSQL database
            conn = psycopg2.connect(
                dbname='chatbot',
                user='postgres',
                password='admin',
                port='5432',
            )
            cur = conn.cursor()

            # Remove the log from the "feedback_logs" table
            delete_query = """
            DELETE FROM chatbot_data
            WHERE id = %s ;
            """
            cur.execute(delete_query, (log_id,))
            conn.commit()
            cur.close()
            conn.close()
            return JsonResponse({'status': 'success'})

        except Exception as e:
            print(f"Error occurred while processing feedback: {e}")
            print(traceback.format_exc())
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
    load_data_from_db()
