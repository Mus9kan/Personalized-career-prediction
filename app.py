from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import psycopg2
from psycopg2 import OperationalError
import openai
import requests
import traceback
import json

app = Flask(__name__)
app.secret_key = 'Harshi@9424'

# ---------------- OpenAI API Key ----------------
openai.api_key = "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"

# ---------------- Database Connection ----------------
def get_db_connection():
    try:
        return psycopg2.connect(
            user="postgres",
            host="localhost",
            database="CareerGuidanceProject",
            password="Harshita@9424",
            port=5432
        )
    except OperationalError as e:
        print("❌ Database connection error:", e)
        return None

# ---------------- Page Routes ----------------
@app.route('/')
def home(): return render_template('home.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/service')
def service(): return render_template('service.html')

@app.route('/course')
def course(): return render_template('course.html')

@app.route('/question')
def question(): return render_template('question.html')

@app.route('/profile')
def profile():
    if 'user' in session:
        return render_template('profile.html', username=session['user'])
    return redirect(url_for('login'))

@app.route('/save-profile', methods=['POST'])
def save_profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    email = session['user']
    full_name = request.form['full_name']
    user_type = request.form['user_type']
    school_details = request.form.get('school_details', '')
    college_details = request.form.get('college_details', '')
    company_role = request.form.get('company_role', '')
    education = request.form.get('education', '')
    experience = request.form.get('experience', '')
    skills = request.form.get('skills', '')

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS UserProfiles (
                email TEXT PRIMARY KEY,
                full_name TEXT,
                user_type TEXT,
                school_details TEXT,
                college_details TEXT,
                company_role TEXT,
                education TEXT,
                experience TEXT,
                skills TEXT
            );
        """)
        cur.execute("""
            INSERT INTO UserProfiles (
                email, full_name, user_type, school_details,
                college_details, company_role, education, experience, skills
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET
                full_name = EXCLUDED.full_name,
                user_type = EXCLUDED.user_type,
                school_details = EXCLUDED.school_details,
                college_details = EXCLUDED.college_details,
                company_role = EXCLUDED.company_role,
                education = EXCLUDED.education,
                experience = EXCLUDED.experience,
                skills = EXCLUDED.skills;
        """, (email, full_name, user_type, school_details, college_details, company_role, education, experience, skills))
        conn.commit()
        return redirect(url_for('profile'))
    except Exception as e:
        conn.rollback()
        return f"Error saving profile: {e}"
    finally:
        cur.close()
        conn.close()

@app.route('/chatbot')
def chatbot():
    if 'user' in session:
        return render_template('chatbot.html', username=session['user'])
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', username=session['user'])
    return redirect(url_for('login'))

# ---------------- Auth Routes ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        if not conn:
            return "Database connection error", 500
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM Users WHERE email = %s AND password = %s", (email, password))
            user = cur.fetchone()
            if user:
                session['user'] = email
                return redirect(url_for('dashboard'))
            return 'Invalid credentials. Please try again.'
        except psycopg2.Error as e:
            return f"Database error during login: {e}", 500
        finally:
            cur.close()
            conn.close()
    return render_template('loginAnimation.html')

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form['email']
    password = request.form['password']
    conn = get_db_connection()
    if not conn:
        return "Database connection error", 500
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM Users WHERE email = %s", (email,))
        if cur.fetchone():
            return 'User already exists! Please login instead.'
        cur.execute("INSERT INTO Users (email, password) VALUES (%s, %s)", (email, password))
        conn.commit()
        session['user'] = email
        return redirect(url_for('dashboard'))
    except psycopg2.Error as e:
        conn.rollback()
        return f"Database error during signup: {e}", 500
    finally:
        cur.close()
        conn.close()

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# ---------------- Conversational LLM Chat ----------------
from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

@app.route('/llm-chat', methods=['POST'])
def llm_chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please type something to begin."})
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a career guidance assistant. Ask aptitude questions one by one."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        print("❌ OpenAI API error:", e)
        traceback.print_exc()
        return jsonify({"reply": f"OpenAI failed: {str(e)}"})

# ---------------- Submit MCQ & Save Prediction History ----------------
@app.route('/submit-test', methods=['POST'])
def submit_test():
    user_answers = request.json.get('answers', [])
    scores = {"abstract": 0, "spatial": 0, "verbal": 0, "perceptual": 0, "numerical": 0}
    difficulty_map = {"easy": 10, "medium": 15, "hard": 20}

    questions = [  # trimmed for brevity... keep yours same
        {"domain": "abstract", "answer": "M", "difficulty": "medium"},
        {"domain": "abstract", "answer": "Apple", "difficulty": "easy"},
        {"domain": "abstract", "answer": "31", "difficulty": "easy"},
        {"domain": "abstract", "answer": "25", "difficulty": "easy"},
        {"domain": "abstract", "answer": "17", "difficulty": "medium"},
        {"domain": "abstract", "answer": "B", "difficulty": "hard"},
        {"domain": "spatial", "answer": "E", "difficulty": "hard"},
        {"domain": "spatial", "answer": "Sphere", "difficulty": "easy"},
        {"domain": "spatial", "answer": "Top", "difficulty": "medium"},
        {"domain": "spatial", "answer": "X", "difficulty": "medium"},
        {"domain": "spatial", "answer": "d", "difficulty": "medium"},
        {"domain": "spatial", "answer": "Circle", "difficulty": "medium"},
        {"domain": "verbal", "answer": "Joyful", "difficulty": "easy"},
        {"domain": "verbal", "answer": "Dull", "difficulty": "medium"},
        {"domain": "verbal", "answer": "Carrot", "difficulty": "easy"},
        {"domain": "verbal", "answer": "Meow", "difficulty": "easy"},
        {"domain": "verbal", "answer": "Cat", "difficulty": "easy"},
        {"domain": "verbal", "answer": "Adverb", "difficulty": "medium"},
        {"domain": "perceptual", "answer": "Rose", "difficulty": "easy"},
        {"domain": "perceptual", "answer": "S", "difficulty": "medium"},
        {"domain": "perceptual", "answer": "Dot", "difficulty": "easy"},
        {"domain": "perceptual", "answer": "ABAB", "difficulty": "easy"},
        {"domain": "perceptual", "answer": "Ǝ", "difficulty": "medium"},
        {"domain": "perceptual", "answer": "C", "difficulty": "hard"},
        {"domain": "numerical", "answer": "30", "difficulty": "easy"},
        {"domain": "numerical", "answer": "32", "difficulty": "medium"},
        {"domain": "numerical", "answer": "48", "difficulty": "easy"},
        {"domain": "numerical", "answer": "5", "difficulty": "easy"},
        {"domain": "numerical", "answer": "7", "difficulty": "easy"},
        {"domain": "numerical", "answer": "24", "difficulty": "medium"}
    ]

    for i, ans in enumerate(user_answers):
        if i >= len(questions):
            break
        q = questions[i]
        if ans.strip().lower() == q["answer"].lower():
            scores[q["domain"]] += difficulty_map[q["difficulty"]]

    try:
        model_response = requests.post("http://localhost:5001/predict", json=scores)
        prediction = model_response.json().get("career", "Unknown")

        if 'user' in session:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS PredictionHistory (
                    id SERIAL PRIMARY KEY,
                    email TEXT,
                    predicted_career TEXT,
                    score_json JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                INSERT INTO PredictionHistory (email, predicted_career, score_json)
                VALUES (%s, %s, %s)
            """, (session['user'], prediction, json.dumps(scores)))
            conn.commit()
            cur.close()
            conn.close()

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({"scores": scores, "career": prediction})

# ---------------- History Display ----------------
@app.route('/prediction-history')
def prediction_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT predicted_career, score_json, timestamp
        FROM PredictionHistory
        WHERE email = %s
        ORDER BY timestamp DESC
        LIMIT 10
    """, (session['user'],))
    history = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('history.html', history=history, username=session['user'])

@app.route('/analytics')
def analytics():
    if 'user' in session:
        return render_template('dashboard.html', username=session['user'])
    return redirect(url_for('login'))

# ---------------- Main ----------------
if __name__ == '__main__':
    app.run(debug=True)
