from flask import Flask, render_template, request, jsonify, send_file, redirect, session
import cv2
import sqlite3
import numpy as np
from datetime import datetime, date
import os
import pandas as pd
from deepface import DeepFace
import base64
from io import BytesIO
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import re
import gc

app = Flask(__name__)

# Generate secret key
SECRET_KEY_FILE = 'secret.key'
if os.path.exists(SECRET_KEY_FILE):
    with open(SECRET_KEY_FILE, 'r') as f:
        app.config['SECRET_KEY'] = f.read().strip()
else:
    app.config['SECRET_KEY'] = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, 'w') as f:
        f.write(app.config['SECRET_KEY'])

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FACES_DIR'] = 'registered_faces'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_DIR'], exist_ok=True)

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

class AttendanceSystem:
    def __init__(self):
        self.db_name = "attendance.db"
        self.faces_dir = app.config['FACES_DIR']
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                organization TEXT,
                created_at TEXT
            )
        ''')
        
        # Persons table with user_id
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                face_image_path TEXT,
                enrollment_date TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, name)
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                date TEXT,
                time TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def enroll_person(self, name, image_path, user_id):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not read image file"
            
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',
                enforce_detection=True,
                align=False
            )
            
            if len(face_objs) == 0:
                return False, "No face detected in the image"
            
            if len(face_objs) > 1:
                return False, "Multiple faces detected. Please use an image with only one face"
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO persons (user_id, name, enrollment_date)
                    VALUES (?, ?, ?)
                ''', (user_id, name, str(date.today())))
                
                person_id = cursor.lastrowid
                
                user_faces_dir = os.path.join(self.faces_dir, f"user_{user_id}")
                os.makedirs(user_faces_dir, exist_ok=True)
                
                dest_path = os.path.join(user_faces_dir, f"{person_id}.jpg")
                
                face_img = face_objs[0]['face']
                face_img = (face_img * 255).astype(np.uint8)
                face_img = cv2.resize(face_img, (224, 224))
                cv2.imwrite(dest_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                cursor.execute('''
                    UPDATE persons SET face_image_path = ? WHERE id = ?
                ''', (dest_path, person_id))
                
                conn.commit()
                
                del face_img
                del face_objs
                gc.collect()
                
                return True, "Person enrolled successfully"
                
            except sqlite3.IntegrityError:
                return False, "Name already exists in your organization"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            gc.collect()
    
    def recognize_faces_in_image(self, image_path, user_id):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, face_image_path FROM persons WHERE user_id = ?', (user_id,))
            persons = cursor.fetchall()
            conn.close()
            
            if not persons:
                return [], "No enrolled persons in your organization"
            
            detected_faces = None
            backend_used = None
            backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
            
            for backend in backends:
                try:
                    detected_faces = DeepFace.extract_faces(
                        img_path=image_path,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=False
                    )
                    if detected_faces and len(detected_faces) > 0:
                        backend_used = backend
                        break
                except Exception as e:
                    continue
            
            if detected_faces is None or len(detected_faces) == 0:
                return [], "No faces detected in image"
            
            recognized_persons = []
            unrecognized_faces = 0
            debug_info = [f"Detector: {backend_used} | Faces found: {len(detected_faces)}\n"]
            
            for face_idx, detected_face in enumerate(detected_faces):
                debug_info.append(f"--- Face {face_idx+1} ---")
                
                temp_face_path = f"temp_face_{face_idx}.jpg"
                try:
                    face_img = detected_face['face']
                    face_img = (face_img * 255).astype(np.uint8)
                    face_img = cv2.resize(face_img, (224, 224))
                    cv2.imwrite(temp_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    
                    best_match = None
                    best_distance = float('inf')
                    best_match_name = None
                    
                    for person_id, name, face_path in persons:
                        if not os.path.exists(face_path):
                            continue
                        
                        try:
                            result = DeepFace.verify(
                                img1_path=temp_face_path,
                                img2_path=face_path,
                                model_name='Facenet512',
                                detector_backend='skip',
                                enforce_detection=False
                            )
                            
                            distance = result['distance']
                            debug_info.append(f"  {name}: {distance:.3f}")
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_match_name = name
                                if distance < 0.40:
                                    best_match = (person_id, name)
                                    
                        except Exception as e:
                            continue
                    
                    if best_match:
                        recognized_persons.append(best_match)
                        debug_info.append(f"  ✓ MATCH: {best_match[1]} ({best_distance:.3f})")
                    else:
                        unrecognized_faces += 1
                        if best_match_name:
                            debug_info.append(f"  ✗ Closest: {best_match_name} ({best_distance:.3f} - too high)")
                        else:
                            debug_info.append(f"  ✗ No match")
                    
                    debug_info.append("")
                    
                    del face_img
                    
                finally:
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
                
                gc.collect()
            
            recognized_persons = list(set(recognized_persons))
            
            full_debug = "\n".join(debug_info)
            full_debug += f"\n{'='*50}\nSummary: {len(detected_faces)} detected, {len(recognized_persons)} recognized, {unrecognized_faces} unrecognized"
            
            return recognized_persons, full_debug
            
        except Exception as e:
            return [], f"Error: {str(e)}"
        finally:
            gc.collect()
    
    def mark_attendance(self, person_ids):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = str(date.today())
        current_time = datetime.now().strftime("%H:%M:%S")
        
        marked = []
        updated = []
        
        for person_id in person_ids:
            cursor.execute('SELECT name FROM persons WHERE id = ?', (person_id,))
            name = cursor.fetchone()[0]
            
            # Check if attendance already exists for today
            cursor.execute('''
                SELECT id FROM attendance 
                WHERE person_id = ? AND date = ?
            ''', (person_id, today))
            
            existing = cursor.fetchone()
            
            if existing:
                # UPDATE existing attendance with new time
                cursor.execute('''
                    UPDATE attendance 
                    SET time = ? 
                    WHERE id = ?
                ''', (current_time, existing[0]))
                updated.append(name)
            else:
                # INSERT new attendance
                cursor.execute('''
                    INSERT INTO attendance (person_id, date, time)
                    VALUES (?, ?, ?)
                ''', (person_id, today, current_time))
                marked.append(name)
        
        conn.commit()
        conn.close()
        
        return marked, updated
    
    def get_attendance_report(self, start_date=None, end_date=None, user_id=None):
        conn = sqlite3.connect(self.db_name)
        
        query = '''
            SELECT p.name, a.date, a.time
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
        '''
        
        conditions = []
        if user_id:
            conditions.append(f"p.user_id = {user_id}")
        if start_date and end_date:
            conditions.append(f"a.date BETWEEN '{start_date}' AND '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY a.date DESC, a.time DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_all_persons(self, user_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT name, enrollment_date FROM persons WHERE user_id = ? ORDER BY name', (user_id,))
        persons = cursor.fetchall()
        conn.close()
        return persons
    
    def delete_person(self, name, user_id):
        conn = None
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, face_image_path FROM persons WHERE name = ? AND user_id = ?', (name, user_id))
            result = cursor.fetchone()
            
            if not result:
                return False, "Person not found or you don't have permission"
            
            person_id, face_path = result
            
            cursor.execute('DELETE FROM attendance WHERE person_id = ?', (person_id,))
            cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
            
            conn.commit()
            
            if face_path and os.path.exists(face_path):
                try:
                    os.remove(face_path)
                except:
                    pass
            
            return True, "Person deleted successfully"
                
        except Exception as e:
            if conn:
                conn.rollback()
            return False, f"Error: {str(e)}"
        finally:
            if conn:
                conn.close()

system = AttendanceSystem()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'message': 'Please log in'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'})
    
    try:
        conn = sqlite3.connect(system.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash, username, organization FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session.clear()
            session['user_id'] = user[0]
            session['username'] = user[2]
            session['organization'] = user[3] if user[3] else ''
            session.permanent = True
            return jsonify({'success': True, 'message': 'Login successful'})
        
        return jsonify({'success': False, 'message': 'Invalid username or password'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    organization = data.get('organization', '').strip()
    
    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All required fields must be filled'})
    
    if len(username) < 3:
        return jsonify({'success': False, 'message': 'Username must be at least 3 characters'})
    
    if not is_valid_email(email):
        return jsonify({'success': False, 'message': 'Please enter a valid email address'})
    
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'})
    
    if not any(c.isupper() for c in password):
        return jsonify({'success': False, 'message': 'Password must contain at least one uppercase letter'})
    
    if not any(c.isdigit() for c in password):
        return jsonify({'success': False, 'message': 'Password must contain at least one number'})
    
    try:
        conn = sqlite3.connect(system.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Username already taken'})
        
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        password_hash = generate_password_hash(password)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, organization, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, organization, str(datetime.now())))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        session.clear()
        session['user_id'] = user_id
        session['username'] = username
        session['organization'] = organization
        session.permanent = True
        
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/user')
@login_required
def get_user():
    return jsonify({
        'username': session.get('username'),
        'organization': session.get('organization', '')
    })

@app.route('/api/enroll', methods=['POST'])
@login_required
def enroll():
    temp_path = None
    try:
        name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image required'})
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_enroll_{os.getpid()}.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        user_id = session.get('user_id')
        success, message = system.enroll_person(name, temp_path, user_id)
        
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        gc.collect()

@app.route('/api/mark-attendance', methods=['POST'])
@login_required
def mark_attendance():
    temp_path = None
    try:
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image required'})
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_attendance_{os.getpid()}.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        user_id = session.get('user_id')
        recognized_persons, debug_info = system.recognize_faces_in_image(temp_path, user_id)
        
        if not recognized_persons:
            return jsonify({
                'success': False, 
                'message': 'No recognized faces',
                'debug': debug_info
            })
        
        marked, updated = system.mark_attendance([p[0] for p in recognized_persons])
        
        return jsonify({
            'success': True,
            'recognized': [name for _, name in recognized_persons],
            'marked': marked,
            'updated': updated,
            'debug': debug_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'debug': f'Error: {str(e)}'})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        gc.collect()

@app.route('/api/persons')
@login_required
def get_persons():
    user_id = session.get('user_id')
    persons = system.get_all_persons(user_id)
    return jsonify({'persons': [{'name': p[0], 'date': p[1]} for p in persons]})

@app.route('/api/attendance')
@login_required
def get_attendance():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    user_id = session.get('user_id')
    
    df = system.get_attendance_report(start_date, end_date, user_id)
    
    return jsonify({
        'records': df.to_dict('records')
    })

@app.route('/api/delete-person', methods=['POST'])
@login_required
def delete_person():
    name = request.json.get('name')
    user_id = session.get('user_id')
    success, message = system.delete_person(name, user_id)
    return jsonify({'success': success, 'message': message})

@app.route('/api/export-csv')
@login_required
def export_csv():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        user_id = session.get('user_id')
        
        df = system.get_attendance_report(start_date, end_date, user_id)
        
        # Check if dataframe is empty
        if df.empty:
            # Return empty CSV with headers
            output = BytesIO()
            output.write(b'name,date,time\n')
            output.seek(0)
        else:
            output = BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'attendance_report_{date.today()}.csv'
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'Export failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎭 AI Attendance Tracker")
    print("=" * 60)
    print("🌐 Server starting at: http://localhost:5000")
    print("📝 Press CTRL+C to stop\n")
    
    app.run(debug=True, host='localhost', port=5000)