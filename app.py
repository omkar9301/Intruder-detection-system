from flask import Flask, request, render_template, redirect, url_for, jsonify, session, flash
import sqlite3
from io import BytesIO
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, url_for, redirect, Response,request, after_this_request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pygame
import imutils
import cv2
import os
import time
from PIL import Image


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for the session

dataset_dir = "D:/intruder detection/name/dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

DATASET_PATH = 'D:/intruder detection/name/dataset'
TRAINER_PATH = 'D:/intruder detection/name/trainer/trainer.yml'
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'


# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              password_hash TEXT,
              email VARCHAR,
              phone_no INTEGER,
              R_address VARCHAR(255),
              gender VARCHAR,
              age INTEGER,
              dob DATE)''')

    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            
            # Hash the password for comparison
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
            user = c.fetchone()
            conn.close()
            
            if user:
                # Store user details in session
                session['user'] = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[3],
                    'phone_no': user[4],
                    'R_address': user[5],
                    'gender': user[6],
                    'age': user[7],
                    'dob': user[8]
                }
                # Redirect to home page
                return redirect('/index')
            else:
                # Invalid credentials, render login page with error message
                return render_template('login1.html', error='Invalid username or password')
        
        except Exception as e:
            # Handle any exceptions
            return render_template('error.html', message="An error occurred during login. Please try again later.")

    # If it's a GET request, render the login page
    return render_template('login1.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            email = request.form['email']
            phone = request.form['phone_no']
            R_address = request.form['R_address']
            gender = request.form['gender']
            age = request.form['age']
            dob = request.form['dob']

            # Hash the password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, password_hash, email, phone, R_address, gender, age, dob))
            conn.commit()
            conn.close()

            return "Registration successful!", 200

        except Exception as e:
            print("Error during registration:", e)
            return "An error occurred during registration. Please try again later.", 500

    return render_template('register1.html')

@app.route('/index',methods=['GET','POST'])
def index():
    # Retrieve user details from session
    user = session.get('user', None)
    if user:
        @after_this_request
        def close_camera(response):
            release_camera()
            return response
        return render_template("index.html", user=user)
    else:
        # Redirect to login page if user is not logged in
        return redirect(url_for('login'))

cap = None

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
#cap=cv2.VideoCapture(0)

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

#cap=cv2.VideoCapture(0)
pygame.init()
pygame.mixer.init()
alarm_sound=pygame.mixer.music.load(os.getcwd()+"/sound.mp3")

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32) 

	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = os.getcwd()+"/face_detector/deploy.prototxt"
weightsPath =os.getcwd()+"/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(os.getcwd()+"/mask_detector.model")


def mask_frames(cap):
    # Counter to avoid saving too many images quickly
    mask_detected_counter = 0

    while True:
        flag, frame = cap.read()
        frame = imutils.resize(frame, width=640, height=480)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            if mask > withoutMask:
                pygame.mixer.music.play(-1)
                label = "Mask"
                color = (0, 255, 0)
                
                # Increment the counter for mask detection
                mask_detected_counter += 1
                
                # Take a screenshot if mask is detected and save it
                if mask_detected_counter == 1:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    file_path = f"screenshots/mask_detected_{timestamp}.png"
                    if not os.path.exists("screenshots"):
                        os.makedirs("screenshots")
                    cv2.imwrite(file_path, frame)
                    
            else:
                pygame.mixer.music.stop()
                label = "No Mask"
                color = (0, 0, 255)
                mask_detected_counter = 0  # Reset counter if no mask

            # Draw label and rectangle on frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = np.asarray(buffer, dtype=np.uint8)
        
        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')



def motion_detection(cap):
    #cap = cv2.VideoCapture(0)
    frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

    out = cv2.VideoWriter("output.avi", fourcc, 5.0, (440,330))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    #print(frame1.shape)
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            
        ret, buffer = cv2.imencode('.jpg', frame1)
        frame1 = buffer.tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+ frame1 +b'\r\n\r\n')
        frame1 = frame2
        ret, frame2 = cap.read()



def name_detection(cap):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.getcwd()+'/name/trainer/trainer.yml')
    cascadePath = os.getcwd()+"/name/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # initiate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1, etc
    names = ['None', 'Omkar']

    # Set video width and height
    cap.set(3, 440)  # Set video width
    cap.set(4, 330)  # Set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cap.get(3)
    minH = 0.1 * cap.get(4)

    while True:
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recognize the face
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 (0 is perfect match)
            if confidence < 100:
                pygame.mixer.music.stop()
                id = names[id]
                confidence_text = "  {0}%".format(round(100 - confidence))
            else:
                pygame.mixer.music.play(-1)
                id = "unknown"
                confidence_text = "  {0}%".format(round(100 - confidence))

                # If "unknown" face is detected, take a screenshot and save it
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_path = f"screenshots/unknown_{timestamp}.png"
                if not os.path.exists("screenshots"):
                    os.makedirs("screenshots")
                cv2.imwrite(file_path, frame)

            # Display the label and confidence on the frame
            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    
 

@app.route('/name', methods=['GET','POST'])
def name():
    return render_template('name.html')

@app.route('/name_video',methods=['GET','POST'])
def name_video():
    return Response(name_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame',)


def visitors_detection(cap):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.getcwd()+'/visitors/trainer/trainer.yml')
    cascadePath = os.getcwd()+"/visitors/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # initiate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1, etc
    names = ['None', 'Visitor']

    # Set video width and height
    cap.set(3, 440)  # Set video width
    cap.set(4, 330)  # Set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cap.get(3)
    minH = 0.1 * cap.get(4)

    while True:
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recognize the face
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 (0 is perfect match)
            if confidence < 100:
                
                id = names[id]
                confidence_text = "  {0}%".format(round(100 - confidence))
            else:
                
                id = 'Unknown'
                confidence_text = "  {0}%".format(round(100 - confidence))

                # If "unknown" face is detected, take a screenshot and save it
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_path = f"screenshots/unknown_{timestamp}.png"
                if not os.path.exists("screenshots"):
                    os.makedirs("screenshots")
                cv2.imwrite(file_path, frame)

            # Display the label and confidence on the frame
            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/visitors', methods=['GET','POST'])
def visitors():
    return render_template('visitors.html')

@app.route('/visitors_video',methods=['GET','POST'])
def visitors_video():
    return Response(visitors_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/',methods=['GET','POST'])
# def first():
#     return redirect(url_for('start'))


@app.route('/mask', methods=['GET'])
def mask():
    return render_template('mask.html')

@app.route('/mask_video', methods=['GET'])
def mask_video():
    return Response(mask_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion', methods=['GET'])
def motion():
    return render_template('motion.html')

@app.route('/motion_video',methods=['GET'])
def motion_video():
    return Response(motion_detection(cap),mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route("/start",methods=["GET", "POST"])
# def start():       
#     return render_template("start.html")

@app.before_request
def before_request():
    init_camera()

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Mock login required decorator for simplicity
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

@app.route('/profile', methods=['GET'])
def profile():
    # Retrieve user details from session
    user = session.get('user', None)
    if user:
        return render_template('profile.html', user=user)
    else:
        # Redirect to login page if user is not logged in
        return redirect(url_for('index'))

@app.route('/facestore')
def facestore():
    return render_template('facestore.html')


@app.route('/capture', methods=['POST'])
def capture_faces():
    face_id = request.form['face_id']
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    print("\n [INFO] Initializing face capture. Look at the camera and wait...")
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(os.path.join(dataset_dir, f"User.{face_id}.{count}.jpg"), gray[y:y + h, x:x + w])

        if count >= 30:  # Take 30 face samples
            break

    cam.release()

    # Redirect to index with a success message
    return redirect(url_for('index', message="Face capture completed! Images captured: {}".format(count)))

@app.route('/facetrain')
def facetrain():
    message = request.args.get('message')
    return render_template('facetrain.html', message=message)

@app.route('/train', methods=['POST'])
def train_faces():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset folder does not exist. Please ensure the dataset is ready."}), 400

    # Ensure opencv-contrib-python is installed
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(HAARCASCADE_PATH)

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('L')  # convert to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return face_samples, ids

    try:
        print("\n [INFO] Training faces. It will take a few seconds. Wait...")
        faces, ids = get_images_and_labels(DATASET_PATH)
        recognizer.train(faces, np.array(ids))

        # Ensure trainer directory exists
        os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)

        # Save the model into trainer/trainer.yml
        recognizer.write(TRAINER_PATH)

        message = "{0} faces trained successfully!".format(len(np.unique(ids)))
        print("\n [INFO] " + message)

        # Redirect to the index page with a success message
        return redirect(url_for('index', message=message))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    # Redirect to the login page
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
