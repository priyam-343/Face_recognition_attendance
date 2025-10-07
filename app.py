from random import randint
from flask import Flask, render_template, request, Response, redirect, send_file,session,url_for
from flask_login import LoginManager,login_required,UserMixin, current_user,login_user,logout_user
from flask_mail import Mail,Message
from flask_sqlalchemy import SQLAlchemy
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
import os
import cv2
import numpy as np
import csv
from deepface import DeepFace
import face_recognition
from datetime import datetime
import timeit
import time
from playsound import playsound
import pandas as pd
import plotly
import plotly.express as px
import json
from dotenv import load_dotenv
import threading

load_dotenv()

app = Flask(__name__)

app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
MAIL_SENDER = os.getenv('MAIL_SENDER')

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///EmployeeDB.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'mysecretkey'
db = SQLAlchemy(app)
mail = Mail(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return users.query.get(user_id)

class employee(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    department = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    hiringDate = db.Column(db.String(10), default=datetime.now().strftime("%d-%m-%Y"))

    def __repr__(self) -> str:
        return f"{self.id} - {self.name} - {self.department} - {self.email} - {self.hiringDate}"

class users(db.Model,UserMixin):
    id = db.Column(db.String(20), primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    name = db.Column(db.String(80), nullable = True) 
    mail = db.Column(db.String(80), nullable = True) 
    password = db.Column(db.String(80), nullable=False)
    # FIX: Corrected column type definition
    dateCreated = db.Column(db.DateTime, default = datetime.utcnow)

    def __repr__(self):
        return '<User {}>'.format(self.username)

path = 'static/TrainingImages'
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

global_recognition_result = {"info_text": "Scanning...", "box_color": (0, 255, 255), "name": ""}
attendance_success_time = 0 
SUCCESS_DISPLAY_DURATION = 5.0

@app.route('/')
def index():
    try:
        cap.release()
    except:
        pass
    try:
        cap2.release()
    except:
        pass
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.query.filter_by(username=username).first()
        if user is not None and user.password == password:
            login_user(user)
            return redirect('/')
        else:
            return render_template('login.html', incorrect=True)
    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect('/')

def send_async_email(app, msg):
    with app.app_context():
        try:
            mail.send(msg)
        except Exception as e:
            print(f"ASYNC MAIL ERROR: {e}")

def send_mail(email, subject, body):
    msg = Message(subject, recipients=[email], sender=MAIL_SENDER, body=body)
    thr = threading.Thread(target=send_async_email, args=[app, msg])
    thr.start()
    return True

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        id = request.form['id']
        username = request.form['username']
        name = request.form['name']
        mail = request.form['mail']
        pass1 = request.form['pass']
        pass2 = request.form['pass2']

        user = users.query.filter_by(username=username).first()
        user2 = users.query.filter_by(id = id).first()
        if user is not None or user2 is not None:
            return render_template('signup.html', incorrect=True, msg = 'User with same ID or Username already exist')
        elif pass1 != pass2:
            return render_template('signup.html', incorrect = True, msg = "Passwords do not match")
        else:
            new_user = users(id = id,name = name, mail = mail,username=username, password=pass1)
            db.session.add(new_user)
            db.session.commit()
            return render_template('login.html', registered = True)

    return render_template('signup.html')


@app.route("/add", methods=['GET', 'POST'])
@login_required
def add():
    try:
        cap2.release()
    except:
        pass
    invalid =0
    if request.method == 'POST':
        id = request.form['id']
        name = request.form['name']
        dept = request.form['dept']
        mail = request.form['mail']
        
        try:
            invalid = 1
            emp = employee(id=id, name=name, department=dept, email=mail)
            db.session.add(emp)
            db.session.commit()
            fileNm = id + '.jpg'
            
            try:
                photo = request.files['photo']
                photo.save(os.path.join(path, fileNm))
            except:
                invalid = 2
                cv2.imwrite(os.path.join(path, fileNm), pic)
                del globals()['pic']
            invalid = 0
        except:
            db.session.rollback()
    allRows = employee.query.all()
    return render_template("insertPage.html", allRows=allRows, invalid = invalid)


@app.route("/delete/<string:id>")
@login_required
def delete(id):
    with app.app_context():
        emp = employee.query.filter_by(id=id).first()
        db.session.delete(emp)
        db.session.commit()
    
    fn = id + ".jpg"
    
    try:
        os.unlink(os.path.join(path, fn))
    except:
        pass
    
    df = pd.read_csv("static/records.csv")
    
    df['Id'] = df['Id'].astype(str)
    
    df.loc[df["Id"] == id, "Status"] = "Terminated"
    
    df.to_csv("static/records.csv", index=False)

    return redirect("/add")

@app.route("/update", methods=['GET', 'POST'])
@login_required
def update():
    id = request.form['id']
    emp = employee.query.filter_by(id=id).first()
    
    emp.name = request.form['name']
    emp.department = request.form['dept']
    emp.email = request.form['mail']
    db.session.commit()
    
    fileNm = id + '.jpg'
    try:
        try:
            photo = request.files['photo']
            photo.save(os.path.join(path, fileNm))
        except:
            cv2.imwrite(os.path.join(path, fileNm), pic)
            del globals()['pic']
    except:
        pass
        
    df = pd.read_csv("static/records.csv")
    df.loc[(df["Id"] == id) & (df['Status'] == 'On Service'), ['Name','Department']] = [emp.name,emp.department]
    df.to_csv("static/records.csv", index=False)
    return redirect("/add")

def gen_frames_takePhoto():
    start = timeit.default_timer()
    captured = False
    countdown_started = False
    countdown_val = 10
    countdown_start_time = None

    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        facesLoc = face_recognition.face_locations(frameS)

        if len(facesLoc) > 1:
            cv2.putText(frame, "Only one person allowed", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif len(facesLoc) == 1 and not captured:
            if not countdown_started:
                countdown_started = True
                countdown_start_time = timeit.default_timer()

            if countdown_started and not captured:
                passed = int(timeit.default_timer() - countdown_start_time)
                remaining = countdown_val - passed

                if remaining >= 0:
                    cv2.putText(frame, str(remaining),
                                (frame.shape[1]//2 - 50, frame.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                else:
                    global pic
                    pic = frame.copy() 
                    playsound("static/cameraSound.wav")
                    captured = True
                    cap2.release()

                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    break

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/takePhoto', methods=['GET', 'POST'])
def takePhoto():
    global cap2
    cap2 = cv2.VideoCapture(0)
    return Response(gen_frames_takePhoto(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/encode")
@login_required
def encode():
    images = []
    myList = os.listdir(path)

    global encodedList
    global imgNames

    def findClassNames(myList):
        cNames = []
        for l in myList:
            curImg = cv2.imread(f'{path}/{l}')
            images.append(curImg)
            cNames.append(os.path.splitext(l)[0])
        return cNames
        
    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            except:
                pass
        return encodeList

    imgNames = findClassNames(myList)
    encodedList = findEncodings(images)
    return render_template("recogPage.html")

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def is_blinking(landmarks):
    leftEye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                        (landmarks.part(37).x, landmarks.part(37).y),
                        (landmarks.part(38).x, landmarks.part(38).y),
                        (landmarks.part(39).x, landmarks.part(39).y),
                        (landmarks.part(40).x, landmarks.part(40).y),
                        (landmarks.part(41).x, landmarks.part(41).y)], np.float32)
    rightEye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                         (landmarks.part(43).x, landmarks.part(43).y),
                         (landmarks.part(44).x, landmarks.part(44).y),
                         (landmarks.part(45).x, landmarks.part(45).y),
                         (landmarks.part(46).x, landmarks.part(46).y),
                         (landmarks.part(47).x, landmarks.part(47).y)], np.float32)
    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    return ear < 0.23

def is_head_moved(landmarks, img_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),        
        (0.0, -330.0, -65.0),   
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), 
        (-150.0, -150.0, -125.0), 
        (150.0, -150.0, -125.0)  
    ])
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    h, w = img_shape[:2]
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return False
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    yaw = angles[1] * 180
    pitch = angles[0] * 180
    return abs(yaw) > 20 or abs(pitch) > 15

def gen_frames():
    global global_recognition_result
    global attendance_success_time
    oldIds = []
    last_processed_time = time.time()
    
    def markEntry(id):
        filepath = 'static/records.csv'
        
        file_is_empty = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        
        with open(filepath, 'a+') as f:
            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            dtime = now.strftime('%H:%M:%S')
            
            with app.app_context():
                emp = employee.query.filter_by(id=id).first()
            
            if emp:
                content = f'{id},{emp.name},{emp.department},{dtime},{date},{"On Service"}'
                
                if file_is_empty:
                    f.writelines(content)
                else:
                    f.writelines(f'\n{content}')

            if emp:
                email_subject = 'Attendance Confirmation: Success'
                email_body = f"""
Hi {emp.name},

We have successfully marked your attendance at {dtime} on {date}.

Employee ID: {id}
Department: {emp.department}

Thank you.
Employee Authentication using Face Recognition Team.
"""
                with app.app_context():
                    send_mail(emp.email, email_subject, email_body)

    process_frame_interval = 0.5 
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        current_time = time.time()

        if current_time - attendance_success_time >= SUCCESS_DISPLAY_DURATION:
            
            imgS_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            facesCurFrame_small = face_recognition.face_locations(imgS_small)
            
            if len(facesCurFrame_small) > 1:
                global_recognition_result = {"info_text": "Only one person allowed", "box_color": (0, 0, 255), "name": ""}
            elif not facesCurFrame_small:
                global_recognition_result = {"info_text": "No Face Detected", "box_color": (255, 255, 255), "name": ""}
            
            if len(facesCurFrame_small) == 1 and (current_time - last_processed_time) > process_frame_interval:
                last_processed_time = current_time
                
                with app.app_context():
                    faceLoc = facesCurFrame_small[0]
                    
                    imgS_rec = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS_rec = cv2.cvtColor(imgS_rec, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    encodesCurFrame = face_recognition.face_encodings(imgS_rec, facesCurFrame_small)

                    if encodesCurFrame:
                        encodeFace = encodesCurFrame[0]
                        matches = face_recognition.compare_faces(encodedList, encodeFace)
                        faceDis = face_recognition.face_distance(encodedList, encodeFace)
                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex] and faceDis[matchIndex] < 0.4:
                            Id = imgNames[matchIndex]
                            emp = employee.query.filter_by(id=Id).first()
                            
                            rect = dlib.rectangle(faceLoc[0] * 4, faceLoc[3] * 4, faceLoc[2] * 4, faceLoc[1] * 4)
                            landmarks = predictor(gray, rect)
                            
                            blinked = is_blinking(landmarks)
                            moved = is_head_moved(landmarks, img.shape)
                            
                            if blinked and moved and Id not in oldIds:
                                markEntry(Id)
                                oldIds.append(Id)
                                attendance_success_time = current_time
                                global_recognition_result = {"info_text": f"Attendance Marked", "box_color": (0, 255, 0), "name": emp.name}
                            elif blinked and moved:
                                global_recognition_result = {"info_text": f"Already Marked", "box_color": (0, 255, 0), "name": emp.name}
                            else:
                                global_recognition_result = {"info_text": "Blink & Move Head", "box_color": (0, 255, 255), "name": emp.name}
                        else:
                            global_recognition_result = {"info_text": 'Unknown', "box_color": (0, 0, 255), "name": ""}
                    else:
                        global_recognition_result = {"info_text": 'Scanning...', "box_color": (0, 255, 255), "name": ""}
        
        
        if len(facesCurFrame_small) == 1:
            faceLoc = facesCurFrame_small[0]
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), global_recognition_result["box_color"], 2)
            
            cv2.putText(img, global_recognition_result["info_text"], (x1, y2 + 25), cv2.FONT_HERSHEY_TRIPLEX, 0.8, global_recognition_result["box_color"], 2)
            if global_recognition_result.get("name"):
                 cv2.putText(img, global_recognition_result["name"], (x1, y2 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, global_recognition_result["box_color"], 2)
        elif len(facesCurFrame_small) >= 1 or not facesCurFrame_small:
            cv2.putText(img, global_recognition_result["info_text"], (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, global_recognition_result["box_color"], 2)


        cv2.putText(img, datetime.now().strftime("%D %H:%M:%S"),
                    (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield(b'--frame\r\n' B'Content-Type: image/jpeg\r\n\r\n'+img+b'\r\n')

            
@app.route('/video', methods=['GET', 'POST'])
def video():
    global cap
    cap = cv2.VideoCapture(0)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/AttendanceSheet")
@login_required
def AttendanceSheet():
    rows = []
    with open('static/records.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    fieldnames = ['Id', 'Name', 'Department', 'Time', 'Date', 'Status']
    return render_template('RecordsPage.html', allrows=rows, fieldnames=fieldnames, len=len)

@app.route("/downloadAll")
def downloadAll():
    return send_file('static/records.csv', as_attachment=True)

@app.route("/downloadToday")
def downloadToday():
    df = pd.read_csv("static/records.csv")
    df = df[df['Date'] == datetime.now().strftime("%d-%m-%Y")]
    df.to_csv("static/todayAttendance.csv", index=False)
    return send_file('static/todayAttendance.csv', as_attachment=True)

@app.route("/resetToday")
@login_required
def resetToday():
    df = pd.read_csv("static/records.csv")
    df = df[df['Date'] != datetime.now().strftime("%d-%m-%Y")]
    df.to_csv("static/records.csv", index=False)
    return redirect('/AttendanceSheet')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=7000)