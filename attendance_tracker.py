import cv2
import face_recognition
import sqlite3
import numpy as np
from datetime import datetime, date
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import pandas as pd

class AttendanceSystem:
    def __init__(self):
        self.db_name = "attendance.db"
        self.encodings_dir = "face_encodings"
        self.init_database()
        self.create_encodings_dir()
        self.known_faces = {}
        self.load_known_faces()
        
    def create_encodings_dir(self):
        """Create directory for storing face encodings"""
        if not os.path.exists(self.encodings_dir):
            os.makedirs(self.encodings_dir)
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT UNIQUE NOT NULL,
                department TEXT,
                enrollment_date TEXT
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                date TEXT,
                time_in TEXT,
                time_out TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id),
                UNIQUE(person_id, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def enroll_person(self, name, employee_id, department, image_path):
        """Enroll a new person with their face"""
        try:
            # Load image and get face encoding
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                return False, "No face detected in the image"
            
            if len(encodings) > 1:
                return False, "Multiple faces detected. Please use an image with only one face"
            
            encoding = encodings[0]
            
            # Save to database
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO persons (name, employee_id, department, enrollment_date)
                    VALUES (?, ?, ?, ?)
                ''', (name, employee_id, department, str(date.today())))
                
                person_id = cursor.lastrowid
                
                # Save encoding to file
                encoding_path = os.path.join(self.encodings_dir, f"{person_id}.npy")
                np.save(encoding_path, encoding)
                
                conn.commit()
                
                # Add to known faces
                self.known_faces[person_id] = {
                    'name': name,
                    'employee_id': employee_id,
                    'encoding': encoding
                }
                
                return True, "Person enrolled successfully"
                
            except sqlite3.IntegrityError:
                return False, "Employee ID already exists"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def load_known_faces(self):
        """Load all enrolled faces from database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, employee_id FROM persons')
        persons = cursor.fetchall()
        conn.close()
        
        for person_id, name, employee_id in persons:
            encoding_path = os.path.join(self.encodings_dir, f"{person_id}.npy")
            if os.path.exists(encoding_path):
                encoding = np.load(encoding_path)
                self.known_faces[person_id] = {
                    'name': name,
                    'employee_id': employee_id,
                    'encoding': encoding
                }
    
    def recognize_face(self, image_path):
        """Recognize a face from an image"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) == 0:
                return None, "No face detected"
            
            if len(face_encodings) > 1:
                return None, "Multiple faces detected"
            
            face_encoding = face_encodings[0]
            
            # Compare with known faces
            for person_id, data in self.known_faces.items():
                match = face_recognition.compare_faces(
                    [data['encoding']], face_encoding, tolerance=0.6
                )
                if match[0]:
                    return person_id, data['name']
            
            return None, "Face not recognized"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def mark_attendance(self, person_id, action="in"):
        """Mark attendance for a person"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = str(date.today())
        current_time = datetime.now().strftime("%H:%M:%S")
        
        try:
            if action == "in":
                # Check if already marked in today
                cursor.execute('''
                    SELECT time_in FROM attendance 
                    WHERE person_id = ? AND date = ?
                ''', (person_id, today))
                
                result = cursor.fetchone()
                if result and result[0]:
                    return False, "Already marked attendance for today"
                
                # Insert or update
                cursor.execute('''
                    INSERT OR REPLACE INTO attendance (person_id, date, time_in)
                    VALUES (?, ?, ?)
                ''', (person_id, today, current_time))
                
                message = f"Attendance marked at {current_time}"
                
            else:  # time out
                cursor.execute('''
                    UPDATE attendance 
                    SET time_out = ?
                    WHERE person_id = ? AND date = ? AND time_out IS NULL
                ''', (current_time, person_id, today))
                
                if cursor.rowcount == 0:
                    return False, "No check-in record found for today"
                
                message = f"Time-out marked at {current_time}"
            
            conn.commit()
            return True, message
            
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            conn.close()
    
    def get_attendance_report(self, start_date=None, end_date=None):
        """Generate attendance report"""
        conn = sqlite3.connect(self.db_name)
        
        query = '''
            SELECT p.name, p.employee_id, p.department, 
                   a.date, a.time_in, a.time_out
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
        '''
        
        if start_date and end_date:
            query += f" WHERE a.date BETWEEN '{start_date}' AND '{end_date}'"
        
        query += " ORDER BY a.date DESC, a.time_in DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_all_persons(self):
        """Get list of all enrolled persons"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT name, employee_id, department FROM persons')
        persons = cursor.fetchall()
        conn.close()
        return persons


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("900x700")
        
        self.system = AttendanceSystem()
        self.camera = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.enroll_tab = ttk.Frame(self.notebook)
        self.attendance_tab = ttk.Frame(self.notebook)
        self.report_tab = ttk.Frame(self.notebook)
        self.view_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.enroll_tab, text="Enroll Person")
        self.notebook.add(self.attendance_tab, text="Mark Attendance")
        self.notebook.add(self.report_tab, text="Reports")
        self.notebook.add(self.view_tab, text="View Enrolled")
        
        self.create_enroll_tab()
        self.create_attendance_tab()
        self.create_report_tab()
        self.create_view_tab()
    
    def create_enroll_tab(self):
        frame = ttk.LabelFrame(self.enroll_tab, text="Enroll New Person", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Name
        ttk.Label(frame, text="Full Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.name_entry = ttk.Entry(frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=5)
        
        # Employee ID
        ttk.Label(frame, text="Employee ID:").grid(row=1, column=0, sticky='w', pady=5)
        self.emp_id_entry = ttk.Entry(frame, width=30)
        self.emp_id_entry.grid(row=1, column=1, pady=5)
        
        # Department
        ttk.Label(frame, text="Department:").grid(row=2, column=0, sticky='w', pady=5)
        self.dept_entry = ttk.Entry(frame, width=30)
        self.dept_entry.grid(row=2, column=1, pady=5)
        
        # Image selection
        ttk.Label(frame, text="Face Photo:").grid(row=3, column=0, sticky='w', pady=5)
        self.image_path_label = ttk.Label(frame, text="No image selected", foreground='gray')
        self.image_path_label.grid(row=3, column=1, sticky='w', pady=5)
        
        ttk.Button(frame, text="Select Image", command=self.select_image).grid(
            row=4, column=1, pady=10, sticky='w'
        )
        
        # Enroll button
        ttk.Button(frame, text="Enroll Person", command=self.enroll_person).grid(
            row=5, column=0, columnspan=2, pady=20
        )
        
        self.selected_image_path = None
    
    def create_attendance_tab(self):
        frame = ttk.LabelFrame(self.attendance_tab, text="Mark Attendance", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(frame, text="Select face photo to mark attendance:").pack(pady=10)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Select Image & Mark IN", 
                  command=lambda: self.mark_attendance_from_image("in")).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Select Image & Mark OUT", 
                  command=lambda: self.mark_attendance_from_image("out")).pack(side='left', padx=5)
        
        # Status label
        self.attendance_status = ttk.Label(frame, text="", font=('Arial', 12))
        self.attendance_status.pack(pady=20)
    
    def create_report_tab(self):
        frame = ttk.Frame(self.report_tab, padding=20)
        frame.pack(fill='both', expand=True)
        
        # Date range selection
        date_frame = ttk.Frame(frame)
        date_frame.pack(pady=10)
        
        ttk.Label(date_frame, text="Start Date (YYYY-MM-DD):").pack(side='left', padx=5)
        self.start_date_entry = ttk.Entry(date_frame, width=15)
        self.start_date_entry.pack(side='left', padx=5)
        
        ttk.Label(date_frame, text="End Date (YYYY-MM-DD):").pack(side='left', padx=5)
        self.end_date_entry = ttk.Entry(date_frame, width=15)
        self.end_date_entry.pack(side='left', padx=5)
        
        ttk.Button(date_frame, text="Generate Report", 
                  command=self.generate_report).pack(side='left', padx=10)
        ttk.Button(date_frame, text="Export to CSV", 
                  command=self.export_report).pack(side='left', padx=5)
        
        # Treeview for report
        columns = ('Name', 'Employee ID', 'Department', 'Date', 'Time In', 'Time Out')
        self.report_tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.report_tree.heading(col, text=col)
            self.report_tree.column(col, width=120)
        
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.report_tree.yview)
        self.report_tree.configure(yscrollcommand=scrollbar.set)
        
        self.report_tree.pack(side='left', fill='both', expand=True, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
    
    def create_view_tab(self):
        frame = ttk.Frame(self.view_tab, padding=20)
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Enrolled Persons", font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Treeview for enrolled persons
        columns = ('Name', 'Employee ID', 'Department')
        self.persons_tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.persons_tree.heading(col, text=col)
            self.persons_tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.persons_tree.yview)
        self.persons_tree.configure(yscrollcommand=scrollbar.set)
        
        self.persons_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        ttk.Button(frame, text="Refresh", command=self.refresh_persons_list).pack(pady=10)
        
        self.refresh_persons_list()
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.selected_image_path = file_path
            self.image_path_label.config(text=os.path.basename(file_path), foreground='black')
    
    def enroll_person(self):
        name = self.name_entry.get().strip()
        emp_id = self.emp_id_entry.get().strip()
        dept = self.dept_entry.get().strip()
        
        if not all([name, emp_id, dept, self.selected_image_path]):
            messagebox.showerror("Error", "Please fill all fields and select an image")
            return
        
        success, message = self.system.enroll_person(name, emp_id, dept, self.selected_image_path)
        
        if success:
            messagebox.showinfo("Success", message)
            # Clear fields
            self.name_entry.delete(0, 'end')
            self.emp_id_entry.delete(0, 'end')
            self.dept_entry.delete(0, 'end')
            self.image_path_label.config(text="No image selected", foreground='gray')
            self.selected_image_path = None
            self.refresh_persons_list()
        else:
            messagebox.showerror("Error", message)
    
    def mark_attendance_from_image(self, action):
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        person_id, result = self.system.recognize_face(file_path)
        
        if person_id is None:
            self.attendance_status.config(text=f"❌ {result}", foreground='red')
            return
        
        success, message = self.system.mark_attendance(person_id, action)
        
        if success:
            self.attendance_status.config(
                text=f"✓ {result}: {message}", 
                foreground='green'
            )
        else:
            self.attendance_status.config(
                text=f"❌ {result}: {message}", 
                foreground='orange'
            )
    
    def generate_report(self):
        start = self.start_date_entry.get().strip()
        end = self.end_date_entry.get().strip()
        
        df = self.system.get_attendance_report(start if start else None, end if end else None)
        
        # Clear existing items
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        # Insert new data
        for _, row in df.iterrows():
            self.report_tree.insert('', 'end', values=tuple(row))
        
        self.current_report_df = df
    
    def export_report(self):
        if not hasattr(self, 'current_report_df'):
            messagebox.showwarning("Warning", "Please generate a report first")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            self.current_report_df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", "Report exported successfully")
    
    def refresh_persons_list(self):
        # Clear existing items
        for item in self.persons_tree.get_children():
            self.persons_tree.delete(item)
        
        # Get and insert persons
        persons = self.system.get_all_persons()
        for person in persons:
            self.persons_tree.insert('', 'end', values=person)


def main():
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()