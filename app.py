from flask import Flask, render_template, request, jsonify, send_file, Response
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

app = Flask(__name__)

# Security: Generate or load secret key
SECRET_KEY_FILE = 'secret.key'
if os.path.exists(SECRET_KEY_FILE):
    with open(SECRET_KEY_FILE, 'r') as f:
        app.config['SECRET_KEY'] = f.read().strip()
else:
    app.config['SECRET_KEY'] = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, 'w') as f:
        f.write(app.config['SECRET_KEY'])

# Encryption key for face data
ENCRYPTION_KEY_FILE = 'encryption.key'
if os.path.exists(ENCRYPTION_KEY_FILE):
    with open(ENCRYPTION_KEY_FILE, 'rb') as f:
        encryption_key = f.read()
else:
    encryption_key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as f:
        f.write(encryption_key)

cipher_suite = Fernet(encryption_key)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FACES_DIR'] = 'registered_faces'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# HTTPS redirect for production
@app.before_request
def before_request():
    if not request.is_secure and os.environ.get('FLASK_ENV') == 'production':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

# ProxyFix for Heroku
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
        self.db_name = "attendance.db"
        self.faces_dir = app.config['FACES_DIR']
        self.encrypted_faces_dir = os.path.join(self.faces_dir, 'encrypted')
        os.makedirs(self.encrypted_faces_dir, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                face_image_path TEXT,
                encrypted_face_path TEXT,
                enrollment_date TEXT
            )
        ''')
        
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
    
    def enroll_person(self, name, image_path):
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
                    INSERT INTO persons (name, enrollment_date)
                    VALUES (?, ?)
                ''', (name, str(date.today())))
                
                person_id = cursor.lastrowid
                
                # Save unencrypted face for processing
                dest_path = os.path.join(self.faces_dir, f"{person_id}.jpg")
                
                face_img = face_objs[0]['face']
                face_img = (face_img * 255).astype(np.uint8)
                face_img = cv2.resize(face_img, (224, 224))
                cv2.imwrite(dest_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                # Encrypt the face image
                encrypted_path = os.path.join(self.encrypted_faces_dir, f"{person_id}.enc")
                encrypted_data = encrypt_file(dest_path)
                
                if encrypted_data:
                    with open(encrypted_path, 'wb') as f:
                        f.write(encrypted_data)
                
                cursor.execute('''
                    UPDATE persons SET face_image_path = ?, encrypted_face_path = ? WHERE id = ?
                ''', (dest_path, encrypted_path, person_id))
                
                conn.commit()
                return True, "Person enrolled successfully (data encrypted)"
                
            except sqlite3.IntegrityError:
                return False, "Name already exists"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def recognize_faces_in_image(self, image_path):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, face_image_path FROM persons')
            persons = cursor.fetchall()
            conn.close()
            
            if not persons:
                return [], "No enrolled persons in database"
            
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
                                if distance < 0.7:
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
    
    def get_attendance_report(self, start_date=None, end_date=None):
        conn = sqlite3.connect(self.db_name)
        
        query = '''
            SELECT p.name, a.date, a.time
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
        '''
        
        if start_date and end_date:
            query += f" WHERE a.date BETWEEN '{start_date}' AND '{end_date}'"
        
        query += " ORDER BY a.date DESC, a.time DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_all_persons(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT name, enrollment_date FROM persons ORDER BY name')
        persons = cursor.fetchall()
        conn.close()
        return persons
    
    def delete_person(self, name):
        conn = None
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, face_image_path FROM persons WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if not result:
                return False, "Person not found"
            
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/enroll', methods=['POST'])
@limiter.limit("10 per hour")  # Limit enrollments
def enroll():
    try:
        name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image required'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save temporary file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_enroll.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        success, message = system.enroll_person(name, temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark-attendance', methods=['POST'])
@limiter.limit("30 per hour")  # Limit attendance marking
def mark_attendance():
    try:
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image required'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save temporary file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_attendance.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        recognized_persons, debug_info = system.recognize_faces_in_image(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if not recognized_persons:
            return jsonify({
                'success': False, 
                'message': 'No recognized faces',
                'debug': debug_info
            })
        
        marked, already_marked = system.mark_attendance([p[0] for p in recognized_persons])
        
        return jsonify({
            'success': True,
            'recognized': [name for _, name in recognized_persons],
            'marked': marked,
            'already_marked': already_marked,
            'debug': debug_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/persons')
def get_persons():
    persons = system.get_all_persons()
    return jsonify({'persons': [{'name': p[0], 'date': p[1]} for p in persons]})

@app.route('/api/attendance')
def get_attendance():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = system.get_attendance_report(start_date, end_date)
    
    return jsonify({
        'records': df.to_dict('records')
    })

@app.route('/api/delete-person', methods=['POST'])
@limiter.limit("20 per hour")  # Limit deletions
def delete_person():
    name = request.json.get('name')
    success, message = system.delete_person(name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/export-csv')
def export_csv():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    df = system.get_attendance_report(start_date, end_date)
    
    # Create CSV in memory
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