import csv
import psycopg2

# Database connection parameters
db_params = {
    'dbname': 'chatbot',
    'user': 'postgres',
    'password': 'admin',  # Add your actual password here
    'port': '5432'  # Add your actual port here, typically 5432 for PostgreSQL
}

# CSV file path
csv_file_path = './dataset.csv'

def insert_data_from_csv():
    # Establish the database connection
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate over each row in the CSV
        for row in reader:
            question = row['question']
            answer = row['answer']
            action = row['action'].split(',')  # Split actions by comma to create a list
            
            # Insert data into the PostgreSQL table
            cur.execute("""
                INSERT INTO chatbot_data (question, answer, actions)
                VALUES (%s, %s, %s)
            """, (question, answer, action))
    
    # Commit the transaction
    conn.commit()
    
    # Close the connection
    cur.close()
    conn.close()
    
    print("Data inserted successfully")

if __name__ == '__main__':
    insert_data_from_csv()
