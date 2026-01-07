from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, session, url_for
import cv2
import sqlite3
import numpy as np
from datetime import datetime, date
import os
import pandas as pd
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import json
import secrets
from cryptography.fernet import Fernet
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import gc
import tempfile

app = Flask(__name__)

# Use temp directory for free tier (no persistent disk)
DATA_DIR = tempfile.gettempdir()
os.makedirs(DATA_DIR, exist_ok=True)

# Security: Generate secret key (will reset on restart, but that's OK for free tier)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Encryption key for face data (will reset on restart)
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')
app.config['FACES_DIR'] = os.path.join(DATA_DIR, 'registered_faces')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Reduced to 10MB

# Session configuration (in-memory, will reset on restart)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False

# HTTPS redirect for production
@app.before_request
def before_request_handler():
    if not request.is_secure and os.environ.get('FLASK_ENV') == 'production':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

# ProxyFix for Render
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_DIR'], exist_ok=True)

class AttendanceSystem:
    def __init__(self):
        self.db_name = os.path.join(DATA_DIR, "attendance.db")
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
        
        # Persons table with user_id AND face embeddings stored
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                face_image_path TEXT,
                face_embedding TEXT,
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
        """Enroll person and store their face embedding for faster comparison"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not read image file"
            
            # Detect faces
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
            
            # Save face image first
            user_faces_dir = os.path.join(self.faces_dir, f"user_{user_id}")
            os.makedirs(user_faces_dir, exist_ok=True)
            
            temp_person_id = f"temp_{os.getpid()}"
            dest_path = os.path.join(user_faces_dir, f"{temp_person_id}.jpg")
            
            # Extract and save face
            face_img = face_objs[0]['face']
            face_img = (face_img * 255).astype(np.uint8)
            face_img = cv2.resize(face_img, (112, 112))  # Even smaller for memory
            cv2.imwrite(dest_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Generate embedding ONCE during enrollment
            try:
                embedding = DeepFace.represent(
                    img_path=dest_path,
                    model_name='Facenet',
                    detector_backend='skip',
                    enforce_detection=False
                )[0]["embedding"]
                
                # Convert to JSON string for storage
                embedding_str = json.dumps(embedding)
            except Exception as e:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False, f"Failed to generate face embedding: {str(e)}"
            
            # Store in database
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO persons (user_id, name, face_embedding, enrollment_date)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, name, embedding_str, str(date.today())))
                
                person_id = cursor.lastrowid
                
                # Rename file to use actual person_id
                final_path = os.path.join(user_faces_dir, f"{person_id}.jpg")
                os.rename(dest_path, final_path)
                
                cursor.execute('''
                    UPDATE persons SET face_image_path = ? WHERE id = ?
                ''', (final_path, person_id))
                
                conn.commit()
                
                # Cleanup
                del face_img
                del face_objs
                del embedding
                gc.collect()
                
                return True, "Person enrolled successfully"
                
            except sqlite3.IntegrityError:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False, "Name already exists in your organization"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            gc.collect()
    
    def recognize_faces_in_image(self, image_path, user_id):
        """OPTIMIZED: Use pre-computed embeddings instead of re-computing"""
        try:
            # Get enrolled persons WITH embeddings
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, face_embedding FROM persons WHERE user_id = ?', (user_id,))
            persons = cursor.fetchall()
            conn.close()
            
            if not persons:
                return [], "No enrolled persons in your organization"
            
            # Parse embeddings from JSON
            enrolled_embeddings = []
            for person_id, name, embedding_str in persons:
                try:
                    embedding = json.loads(embedding_str)
                    enrolled_embeddings.append((person_id, name, np.array(embedding)))
                except:
                    continue
            
            if not enrolled_embeddings:
                return [], "No valid face embeddings found"
            
            # Detect faces using ONLY opencv (lightest detector)
            try:
                detected_faces = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend='opencv',
                    enforce_detection=False,
                    align=False
                )
            except Exception as e:
                return [], f"Face detection failed: {str(e)}"
            
            if not detected_faces or len(detected_faces) == 0:
                return [], "No faces detected in image. Try a clearer photo with better lighting."
            
            recognized_persons = []
            unrecognized_faces = 0
            debug_info = [f"Using opencv detector - Found {len(detected_faces)} face(s)\n"]
            
            # Limit to first 3 faces to avoid memory issues
            detected_faces = detected_faces[:3]
            debug_info.append(f"Processing first {len(detected_faces)} face(s) to save memory\n")
            
            # Process each detected face
            for face_idx, detected_face in enumerate(detected_faces):
                debug_info.append(f"--- Face {face_idx+1} ---")
                
                temp_face_path = os.path.join(DATA_DIR, f'temp_face_{os.getpid()}.jpg')
                try:
                    # Save detected face
                    face_img = detected_face['face']
                    face_img = (face_img * 255).astype(np.uint8)
                    face_img = cv2.resize(face_img, (112, 112))
                    cv2.imwrite(temp_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    
                    # Generate embedding ONCE for this face
                    try:
                        detected_embedding = DeepFace.represent(
                            img_path=temp_face_path,
                            model_name='Facenet',
                            detector_backend='skip',
                            enforce_detection=False
                        )[0]["embedding"]
                        detected_embedding = np.array(detected_embedding)
                    except Exception as e:
                        debug_info.append(f"  Error generating embedding: {str(e)}")
                        continue
                    
                    best_match = None
                    best_distance = float('inf')
                    best_match_name = None
                    
                    # Compare with pre-computed embeddings (MUCH faster and less memory)
                    for person_id, name, enrolled_embedding in enrolled_embeddings:
                        try:
                            # Calculate cosine distance manually
                            distance = np.linalg.norm(detected_embedding - enrolled_embedding)
                            
                            debug_info.append(f"  {name}: {distance:.3f}")
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_match_name = name
                                if distance < 10.0:  # Threshold for euclidean distance
                                    best_match = (person_id, name)
                        except Exception as e:
                            continue
                    
                    # Report result
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
                    
                    # Cleanup
                    del face_img
                    del detected_embedding
                    
                finally:
                    if os.path.exists(temp_face_path):
                        try:
                            os.remove(temp_face_path)
                        except:
                            pass
                
                # Force GC after each face
                gc.collect()
            
            # Remove duplicates
            recognized_persons = list(set(recognized_persons))
            
            full_debug = "\n".join(debug_info)
            full_debug += f"\nSummary: {len(detected_faces)} face(s) detected, {len(recognized_persons)} recognized, {unrecognized_faces} unrecognized"
            
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
        already_marked = []
        
        for person_id in person_ids:
            cursor.execute('''
                SELECT COUNT(*) FROM attendance 
                WHERE person_id = ? AND date = ?
            ''', (person_id, today))
            
            count = cursor.fetchone()[0]
            
            cursor.execute('SELECT name FROM persons WHERE id = ?', (person_id,))
            name = cursor.fetchone()[0]
            
            if count == 0:
                cursor.execute('''
                    INSERT INTO attendance (person_id, date, time)
                    VALUES (?, ?, ?)
                ''', (person_id, today, current_time))
                marked.append(name)
            else:
                already_marked.append(name)
        
        conn.commit()
        conn.close()
        
        return marked, already_marked
    
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

# Initialize system
system = AttendanceSystem()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Please log in', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/dashboard')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('dashboard.html')

# Authentication routes
@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'})
    
    try:
        conn = sqlite3.connect(system.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash, username, organization FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = user[2]
            session['organization'] = user[3]
            return jsonify({'success': True, 'message': 'Login successful'})
        
        return jsonify({'success': False, 'message': 'Invalid username or password'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/register', methods=['POST'])
@limiter.limit("3 per hour")
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    organization = data.get('organization', '')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All fields required'})
    
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'})
    
    try:
        conn = sqlite3.connect(system.db_name)
        cursor = conn.cursor()
        
        password_hash = generate_password_hash(password)
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, organization, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, organization, str(datetime.now())))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        session['user_id'] = user_id
        session['username'] = username
        session['organization'] = organization
        
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return jsonify({'success': False, 'message': 'Username already exists'})
        elif 'email' in str(e):
            return jsonify({'success': False, 'message': 'Email already exists'})
        return jsonify({'success': False, 'message': 'Registration failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/user')
@login_required
def get_user():
    return jsonify({
        'username': session.get('username'),
        'organization': session.get('organization')
    })

@app.route('/api/enroll', methods=['POST'])
@limiter.limit("10 per hour")
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
@limiter.limit("30 per hour")
@login_required
def mark_attendance():
    temp_path = None
    try:
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image required', 'debug': 'No image data received'})
        
        try:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'success': False, 'message': 'Invalid image data', 'debug': str(e)})
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_attendance_{os.getpid()}.jpg')
        try:
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            return jsonify({'success': False, 'message': 'Failed to save image', 'debug': str(e)})
        
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'message': 'Image file not found', 'debug': 'File save failed'})
        
        user_id = session.get('user_id')
        
        try:
            recognized_persons, debug_info = system.recognize_faces_in_image(temp_path, user_id)
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': 'Recognition failed',
                'debug': f'Error during recognition: {str(e)}'
            })
        
        if not recognized_persons:
            return jsonify({
                'success': False, 
                'message': 'No recognized faces',
                'debug': debug_info
            })
        
        try:
            marked, already_marked = system.mark_attendance([p[0] for p in recognized_persons])
        except Exception as e:
            return jsonify({
                'success': False,
                'message': 'Failed to mark attendance',
                'debug': f'Database error: {str(e)}'
            })
        
        return jsonify({
            'success': True,
            'recognized': [name for _, name in recognized_persons],
            'marked': marked,
            'already_marked': already_marked,
            'debug': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': 'Server error',
            'debug': f'Unexpected error: {str(e)}'
        })
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
@limiter.limit("20 per hour")
@login_required
def delete_person():
    name = request.json.get('name')
    user_id = session.get('user_id')
    success, message = system.delete_person(name, user_id)
    return jsonify({'success': success, 'message': message})

@app.route('/api/export-csv')
@login_required
def export_csv():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    user_id = session.get('user_id')
    
    df = system.get_attendance_report(start_date, end_date, user_id)
    
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_report_{date.today()}.csv'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)