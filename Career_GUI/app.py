from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_socketio import join_room, leave_room, send, SocketIO
import random
from string import ascii_uppercase
import joblib
import psycopg2
import numpy as np
import pandas as pd


app = Flask(__name__)
app.secret_key = 'career'
app.config["SECRET_KEY"] = "shree123"
socketio = SocketIO(app)

# Load the trained model
Yield1 = joblib.load('models/svm_c1.pkl')
Yield2 = joblib.load('models/preprocessor_c1.pkl')

high1 = joblib.load('models/scaler1.pkl')
high2 = joblib.load('models/model1.pkl')


#generates unique code for community chat
rooms = {}
def generate_unique_code(length):
    while True:
        code = ""
        for _ in range(length):
            code += random.choice(ascii_uppercase)

        if code not in rooms:
            break
    return code



#Career Guidance after PUC
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']


def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,
                               total_score, average_score]])

    # Scale features
    scaled_features = high1.transform(feature_array)

    # Predict using the model
    predicted_index = high2.predict(scaled_features)[0]

    # Get class name from predicted index
    predicted_class_name = class_names[predicted_index]

    return predicted_class_name


#Connecting to database
def connect_to_db():
    return psycopg2.connect(
        user="postgres",
        password="sql@2024",
        host="localhost",
        port="5432",
        database="Career"
    )

conn = connect_to_db()
cursor = conn.cursor()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/add_users', methods=['POST'])
def add_users():
    name = request.form.get('uname')
    email = request.form.get('uemail')
    password = request.form.get('upassword')

    cursor.execute("""
        INSERT INTO login (name, email, password)
        VALUES (%s, %s, %s)
    """, (name, email, password))
    conn.commit()

    session['user_id'] = cursor.lastrowid  # assuming `id` is auto-incremented
    session['user_name'] = name
    session['user_email'] = email

    return render_template("successful.html")

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("SELECT user_id, name, email FROM login WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()

    if user:
        session['user_id'] = user[0]
        session['user_name'] = user[1]
        session['user_email'] = user[2]
        return redirect('/starter')
    else:
        flash('Invalid email or password', 'danger')
        return redirect('/')

@app.route('/starter')
def starter():
    name = session.get('user_name')
    if name:
        return render_template("new.html", name=name)
    else:
        flash('Please log in first.', 'warning')
        return redirect('/')

@app.route('/profile')
def profile():
    email = session.get('user_email')
    if email:
        cursor.execute("""
            SELECT name, email, predicted_career, course_rating, review
            FROM login
            WHERE email = %s
        """, (email,))
        user_info = cursor.fetchone()
        return render_template('profile.html', user_info=user_info)
    else:
        flash('Please log in first.', 'warning')
        return redirect('/')

@app.route('/submit_review', methods=['POST'])
def submit_review():
    course_rating = request.form.get('course_rating')
    review = request.form.get('course_review')
    user_email = session.get('user_email')

    cursor.execute("""
        UPDATE login
        SET course_rating = %s, review = %s
        WHERE email = %s
    """, (course_rating, review, user_email))
    conn.commit()

    flash('Your review has been submitted successfully!', 'success')
    return redirect('/profile')


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form.get('part_time_job') == 'on'
        absence_days = int(request.form['absence_days'])
        extracurricular_activities = request.form.get('extracurricular_activities') == 'on'
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        # Get recommendation
        recommendation = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                         weekly_self_study_hours, math_score, history_score, physics_score,
                                         chemistry_score, biology_score, english_score, geography_score,
                                         total_score, average_score)

        return render_template('result1.html', recommendation=recommendation)
    else:
        return render_template('high.html')


@app.route('/predict', methods=['GET', 'POST'])
def yield1():
    name = session.get('user_name')
    if request.method == 'POST':
        Logical_quotient_rating = request.form['Logical_quotient_rating']
        hackathons = request.form['hackathons']
        coding_skills_rating = request.form['coding_skills_rating']
        public_speaking_points = request.form['public_speaking_points']
        self_learning_capability = request.form['self_learning_capability']
        Extra_courses_did = request.form['Extra_courses_did']
        certifications = request.form['certifications']
        workshops = request.form['workshops']
        reading_and_writing_skills = request.form['reading_and_writing_skills']
        memory_capability_score = request.form['memory_capability_score']
        Interested_subjects = request.form['Interested_subjects']
        interested_career_area = request.form['interested_career_area']
        Type_of_company_want_to_settle = request.form['Type_of_company_want_to_settle']
        Taken_inputs_from_seniors_or_elders = request.form['Taken_inputs_from_seniors_or_elders']
        Management_or_Technical = request.form['Management_or_Technical']
        Interested_Type_of_Books = request.form['Interested_Type_of_Books']
        hard_or_smart_worker = request.form['hard_or_smart_worker']
        worked_in_teams_ever = request.form['worked_in_teams_ever']
        Introvert = request.form['Introvert']

        column_names = ['Logical_quotient_rating', 'hackathons', 'coding_skills_rating', 'public_speaking_points',
                        'self_learning_capability', 'Extra_courses_did', 'certifications', 'workshops',
                        'reading_and_writing_skills', 'memory_capability_score', 'Interested_subjects',
                        'interested_career_area', 'Type_of_company_want_to_settle',
                        'Taken_inputs_from_seniors_or_elders', 'Management_or_Technical', 'Interested_Type_of_Books',
                        'hard_or_smart_worker', 'worked_in_teams_ever', 'Introvert']

        features = pd.DataFrame([[Logical_quotient_rating, hackathons, coding_skills_rating, public_speaking_points,
                                  self_learning_capability, Extra_courses_did, certifications, workshops,
                                  reading_and_writing_skills, memory_capability_score, Interested_subjects,
                                  interested_career_area, Type_of_company_want_to_settle,
                                  Taken_inputs_from_seniors_or_elders, Management_or_Technical,
                                  Interested_Type_of_Books, hard_or_smart_worker, worked_in_teams_ever, Introvert]],
                                columns=column_names)

        transformed_features = Yield2.transform(features)

        predicted = Yield1.predict(transformed_features)
        career = predicted[0]
        session['predicted'] = career

        user_email = session.get('user_email')
        cursor.execute("""
            UPDATE login
            SET predicted_career = %s
            WHERE email = %s
        """, (career, user_email))
        conn.commit()

        return render_template('result.html', career=career, name=name)
    else:
        return render_template('yield.html')

@app.route('/roadmap/<career>')
def roadmap(career):
    career_templates = {
        "CRM Technical Developer": "CRM.html",
        "Web Developer": "web.html",
        "Network Security Engineer" : "NSE.html",
        "Software Engineer": "SE.html",
        "UX Designer": "UD.html",
        "Software Developer": "SD.html",
        "Database Developer": "DD.html",
        "Software Quality Assurance (QA) Testing": "QA.html",
        "Technical Support": "TS.html",
        "Systems Security Administrator": "SSA.html",
        "Applications Developer": "AD.html",
        "Mobile Applications Developer": "MAD.html"
    }

    template_file = career_templates.get(career)
    if template_file:
        return render_template(template_file, career=career)
    else:
        return "No Roadmap available"


@app.route('/chat', methods=["POST", "GET"])
def chat():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        code = request.form.get("code")
        join = request.form.get("join", False)
        create = request.form.get("create", False)

        if not name:
            return render_template("home.html", error="Please enter a name.", code=code, name=name)

        if join != False and not code:
            return render_template("home.html", error="Please enter a room code.", code=code, name=name)

        room = code
        if create != False:
            room = generate_unique_code(4)
            rooms[room] = {"members" : 0, "messages" : []}
        elif code not in rooms:
            return render_template("home.html", error="Room does not exist.", code=code, name=name)

        session["room"] = room
        session["name"] = name

        return redirect(url_for("room"))
    return render_template("home.html")


@app.route("/room")
def room():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))
    return render_template("room.html", code=room, messages=rooms[room]["messages"])

@socketio.on("message")
def message(data):
    room =session.get("room")
    if room not in rooms:
        return

    content = {
        "name": session.get("name"),
        "message": data["data"]
    }
    send(content, to=room)
    rooms[room]["messages"].append(content)
    print(f"{session.get('name')} said: {data['data']}")


@socketio.on("connect")
def connect(auth):
    room = session.get("room")
    name = session.get("name")

    if not room or not name:
        return
    if room not in rooms:
        leave_room(room)
        return

    join_room(room)
    send({"name": name, "message": "has entered the room"}, to=room)
    rooms[room]["members"] += 1
    print(f"{name} joined the room {room}")



@socketio.on("disconnect")
def disconnect():
    room = session.get("room")
    name = session.get("name")
    leave_room(room)

    if room in rooms:
        rooms[room]['members'] -= 1
        if rooms[room]["members"] <= 0:
            del rooms[room]
    send({"name": name, "message": "has entered the room"}, to=room)
    print(f"{name} has left the {room}")


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    socketio.run(app, debug=True,allow_unsafe_werkzeug=True, port=5000)
