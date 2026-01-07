from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, session, url_for
import cv2
import sqlite3
import numpy as np
from datetime import datetime, date
import os
import pandas as pd
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
import gc  # Garbage collection

app = Flask(__name__)

# Use persistent disk on Render, fallback to local for development
DATA_DIR = os.environ.get('RENDER') and '/opt/render/project/data' or '.'
os.makedirs(DATA_DIR, exist_ok=True)

# Security: Generate or load secret key
SECRET_KEY_FILE = os.path.join(DATA_DIR, 'secret.key')
if os.path.exists(SECRET_KEY_FILE):
    with open(SECRET_KEY_FILE, 'r') as f:
        app.config['SECRET_KEY'] = f.read().strip()
else:
    app.config['SECRET_KEY'] = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, 'w') as f:
        f.write(app.config['SECRET_KEY'])

# Encryption key for face data
ENCRYPTION_KEY_FILE = os.path.join(DATA_DIR, 'encryption.key')
if os.path.exists(ENCRYPTION_KEY_FILE):
    with open(ENCRYPTION_KEY_FILE, 'rb') as f:
        encryption_key = f.read()
else:
    encryption_key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as f:
        f.write(encryption_key)

cipher_suite = Fernet(encryption_key)

app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')
app.config['FACES_DIR'] = os.path.join(DATA_DIR, 'registered_faces')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

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

def encrypt_file(file_path):
    """Encrypt a file and return encrypted data"""
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        encrypted_data = cipher_suite.encrypt(file_data)
        return encrypted_data
    except Exception as e:
        print(f"Encryption error: {e}")
        return None

def decrypt_file(encrypted_data, output_path):
    """Decrypt data and save to file"""
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        return True
    except Exception as e:
        print(f"Decryption error: {e}")
        return False

class AttendanceSystem:
    def __init__(self):
        self.db_name = os.path.join(DATA_DIR, "attendance.db")
        self.faces_dir = app.config['FACES_DIR']
        self.encrypted_faces_dir = os.path.join(self.faces_dir, 'encrypted')
        os.makedirs(self.encrypted_faces_dir, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Users table for authentication
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
        
        # Persons table with user_id for data isolation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                face_image_path TEXT,
                encrypted_face_path TEXT,
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
            
            # Use OpenCV's face detector (no heavy models needed)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return False, "No face detected in the image"
            
            if len(faces) > 1:
                return False, "Multiple faces detected. Please use an image with only one face"
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO persons (user_id, name, enrollment_date)
                    VALUES (?, ?, ?)
                ''', (user_id, name, str(date.today())))
                
                person_id = cursor.lastrowid
                
                # Save face image (in user-specific folder)
                user_faces_dir = os.path.join(self.faces_dir, f"user_{user_id}")
                os.makedirs(user_faces_dir, exist_ok=True)
                
                # Extract and save face region
                x, y, w, h = faces[0]
                face_roi = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                
                dest_path = os.path.join(user_faces_dir, f"{person_id}.jpg")
                cv2.imwrite(dest_path, face_resized)
                
                # Encrypt the face image
                encrypted_path = os.path.join(self.encrypted_faces_dir, f"user_{user_id}_{person_id}.enc")
                encrypted_data = encrypt_file(dest_path)
                
                if encrypted_data:
                    with open(encrypted_path, 'wb') as f:
                        f.write(encrypted_data)
                
                cursor.execute('''
                    UPDATE persons SET face_image_path = ?, encrypted_face_path = ? WHERE id = ?
                ''', (dest_path, encrypted_path, person_id))
                
                conn.commit()
                return True, "Person enrolled successfully"
                
            except sqlite3.IntegrityError:
                return False, "Name already exists in your organization"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def recognize_faces_in_image(self, image_path, user_id):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, face_image_path FROM persons WHERE user_id = ?', (user_id,))
            persons = cursor.fetchall()
            conn.close()
            
            if not persons:
                return [], "No enrolled persons in your organization"
            
            # Load test image
            img = cv2.imread(image_path)
            if img is None:
                return [], "Could not read image file"
            
            # Detect faces using OpenCV (lightweight, no model download)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(detected_faces) == 0:
                return [], "No faces detected in image"
            
            recognized_persons = []
            debug_info = [f"Detector: opencv-haar | Faces found: {len(detected_faces)}\n"]
            
            # Create ORB descriptor for feature matching
            orb = cv2.ORB_create()
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            for face_idx, (x, y, w, h) in enumerate(detected_faces):
                debug_info.append(f"--- Face {face_idx+1} ---")
                
                # Extract face region
                face_roi = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                
                # Get keypoints and descriptors
                kp1, des1 = orb.detectAndCompute(face_gray, None)
                
                if des1 is None:
                    debug_info.append("  ✗ No features detected")
                    continue
                
                best_match = None
                best_match_count = 0
                
                for person_id, name, face_path in persons:
                    if not os.path.exists(face_path):
                        continue
                    
                    try:
                        # Load stored face
                        stored_face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                        if stored_face is None:
                            continue
                        
                        # Get keypoints and descriptors
                        kp2, des2 = orb.detectAndCompute(stored_face, None)
                        
                        if des2 is None:
                            continue
                        
                        # Match descriptors
                        matches = bf.match(des1, des2)
                        match_count = len(matches)
                        
                        # Calculate match quality
                        good_matches = [m for m in matches if m.distance < 50]
                        match_quality = len(good_matches)
                        
                        debug_info.append(f"  {name}: {match_quality} good matches")
                        
                        # Threshold: at least 15 good matches
                        if match_quality > 15 and match_quality > best_match_count:
                            best_match_count = match_quality
                            best_match = (person_id, name)
                    
                    except Exception as e:
                        continue
                
                if best_match:
                    recognized_persons.append(best_match)
                    debug_info.append(f"  ✓ MATCH: {best_match[1]} ({best_match_count} matches)")
                else:
                    debug_info.append(f"  ✗ No match (best: {best_match_count} matches)")
                
                debug_info.append("")
            
            recognized_persons = list(set(recognized_persons))
            
            full_debug = "\n".join(debug_info)
            full_debug += f"\nSummary: {len(detected_faces)} detected, {len(recognized_persons)} recognized"
            
            return recognized_persons, full_debug
            
        except Exception as e:
            return [], f"Error: {str(e)}"
                person_id = cursor.lastrowid
                
                # Save unencrypted face for processing (in user-specific folder)
                user_faces_dir = os.path.join(self.faces_dir, f"user_{user_id}")
                os.makedirs(user_faces_dir, exist_ok=True)
                
                dest_path = os.path.join(user_faces_dir, f"{person_id}.jpg")
                
                face_img = face_objs[0]['face']
                face_img = (face_img * 255).astype(np.uint8)
                face_img = cv2.resize(face_img, (224, 224))
                cv2.imwrite(dest_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                # Encrypt the face image
                encrypted_path = os.path.join(self.encrypted_faces_dir, f"user_{user_id}_{person_id}.enc")
                encrypted_data = encrypt_file(dest_path)
                
                if encrypted_data:
                    with open(encrypted_path, 'wb') as f:
                        f.write(encrypted_data)
                
                cursor.execute('''
                    UPDATE persons SET face_image_path = ?, encrypted_face_path = ? WHERE id = ?
                ''', (dest_path, encrypted_path, person_id))
                
                conn.commit()
                return True, "Person enrolled successfully"
                
            except sqlite3.IntegrityError:
                return False, "Name already exists in your organization"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def recognize_faces_in_image(self, image_path, user_id):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, face_image_path FROM persons WHERE user_id = ?', (user_id,))
            persons = cursor.fetchall()
            conn.close()
            
            if not persons:
                return [], "No enrolled persons in your organization"
            
            # Use only OpenCV detector (fastest and lightest)
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
                return [], "No faces detected in image"
            
            recognized_persons = []
            debug_info = [f"Detector: opencv | Faces found: {len(detected_faces)}\n"]
            
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
                    
                    for person_id, name, face_path in persons:
                        if not os.path.exists(face_path):
                            continue
                        
                        try:
                            # Use Facenet (lighter than Facenet512)
                            result = DeepFace.verify(
                                img1_path=temp_face_path,
                                img2_path=face_path,
                                model_name='Facenet',  # Changed from Facenet512
                                detector_backend='skip',
                                enforce_detection=False
                            )
                            
                            distance = result['distance']
                            debug_info.append(f"  {name}: {distance:.3f}")
                            
                            if distance < best_distance:
                                best_distance = distance
                                if distance < 0.4:  # Adjusted threshold for Facenet
                                    best_match = (person_id, name)
                                    
                        except Exception as e:
                            continue
                    
                    if best_match:
                        recognized_persons.append(best_match)
                        debug_info.append(f"  ✓ MATCH: {best_match[1]} ({best_distance:.3f})")
                    else:
                        debug_info.append(f"  ✗ No match ({best_distance:.3f})")
                    
                    debug_info.append("")
                    
                finally:
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
            
            recognized_persons = list(set(recognized_persons))
            
            full_debug = "\n".join(debug_info)
            full_debug += f"\nSummary: {len(detected_faces)} detected, {len(recognized_persons)} recognized"
            
            return recognized_persons, full_debug
            
        except Exception as e:
            return [], f"Error: {str(e)}"
    
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

# Landing page
@app.route('/')
def landing():
    return render_template('landing.html')

# Login page
@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/dashboard')
    return render_template('login.html')

# Dashboard (main app)
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

# Attendance routes
@app.route('/api/enroll', methods=['POST'])
@limiter.limit("10 per hour")
@login_required
def enroll():
    try:
        name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image required'})
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_enroll.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        user_id = session.get('user_id')
        success, message = system.enroll_person(name, temp_path, user_id)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark-attendance', methods=['POST'])
@limiter.limit("30 per hour")
@login_required
def mark_attendance():
    temp_path = None
    try:
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image required', 'debug': 'No image data received'})
        
        # Decode image - limit size to prevent memory issues
        try:
            # Split and get base64 data
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Check base64 size (rough estimate: 4/3 of actual size)
            estimated_size = len(image_data) * 3 / 4
            if estimated_size > 5 * 1024 * 1024:  # 5MB limit
                return jsonify({
                    'success': False, 
                    'message': 'Image too large (max 5MB)',
                    'debug': f'Estimated size: {estimated_size/1024/1024:.1f}MB'
                })
            
            image_bytes = base64.b64decode(image_data)
            
            # Clear the base64 string from memory immediately
            del image_data
            gc.collect()
            
        except Exception as e:
            return jsonify({'success': False, 'message': 'Invalid image data', 'debug': str(e)})
        
        # Convert to PIL Image and resize to max 800x800 to save memory
        try:
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Resize if too large
            max_size = 800
            if pil_image.width > max_size or pil_image.height > max_size:
                ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Clear original bytes from memory
            del image_bytes
            gc.collect()
            
        except Exception as e:
            return jsonify({'success': False, 'message': 'Invalid image format', 'debug': str(e)})
        
        # Save temporary file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_att_{datetime.now().timestamp()}.jpg')
        try:
            pil_image.save(temp_path, 'JPEG', quality=85, optimize=True)
            
            # Clear PIL image from memory
            del pil_image
            gc.collect()
            
        except Exception as e:
            return jsonify({'success': False, 'message': 'Failed to save image', 'debug': str(e)})
        
        # Verify file exists
        if not os.path.exists(temp_path):
            return jsonify({'success': False, 'message': 'Image file not found', 'debug': 'File save failed'})
        
        user_id = session.get('user_id')
        
        # Recognize faces
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
        
        # Mark attendance
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
        # Always cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        # Force garbage collection
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