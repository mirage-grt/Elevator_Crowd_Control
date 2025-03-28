from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize camera
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    if 'role' not in session:
        return redirect(url_for('login', role='user'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'role' not in session:
        return redirect(url_for('login', role='user'))
    return render_template('dashboard.html')

@app.route('/login/<role>', methods=['GET', 'POST'])
def login(role):
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if role == 'admin' and username == 'admin' and password == 'admin':
            session['role'] = 'admin'
            return redirect(url_for('dashboard'))
        elif role == 'user' and username == 'user' and password == 'user':
            session['role'] = 'user'
            return redirect(url_for('dashboard'))
        
        return render_template('login.html', role=role, error='Invalid credentials')

    session.clear()
    return render_template('login.html', role=role)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login', role='user'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
