import cv2
import sqlite3
import numpy as np
from datetime import datetime, date
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import pandas as pd
from deepface import DeepFace

class AttendanceSystem:
    def __init__(self):
        self.db_name = "attendance.db"
        self.faces_dir = "registered_faces"
        self.init_database()
        self.create_faces_dir()
        
    def create_faces_dir(self):
        """Create directory for storing face images"""
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                face_image_path TEXT,
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
        """Enroll a new person with their face"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not read image file"
            
            # Detect faces WITHOUT alignment to preserve orientation
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',
                enforce_detection=True,
                align=False  # Don't rotate the face
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
                
                # Save the face without rotation
                dest_path = os.path.join(self.faces_dir, f"{person_id}.jpg")
                
                # Get the face (not aligned/rotated)
                face_img = face_objs[0]['face']
                face_img = (face_img * 255).astype(np.uint8)
                
                # Resize to standard size for consistency
                face_img = cv2.resize(face_img, (224, 224))
                
                # Save as high quality JPEG
                cv2.imwrite(dest_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                cursor.execute('''
                    UPDATE persons SET face_image_path = ? WHERE id = ?
                ''', (dest_path, person_id))
                
                conn.commit()
                return True, "Person enrolled successfully"
                
            except sqlite3.IntegrityError:
                return False, "Name already exists"
            finally:
                conn.close()
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def recognize_faces_in_image(self, image_path):
        """Recognize all faces in an image (works with group photos)"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, face_image_path FROM persons')
            persons = cursor.fetchall()
            conn.close()
            
            if not persons:
                return [], "No enrolled persons in database"
            
            # Try multiple detection backends for better detection
            detected_faces = None
            backend_used = None
            backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
            
            for backend in backends:
                try:
                    detected_faces = DeepFace.extract_faces(
                        img_path=image_path,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=False  # Don't rotate faces during detection
                    )
                    if detected_faces and len(detected_faces) > 0:
                        backend_used = backend
                        break
                except Exception as e:
                    continue
            
            if detected_faces is None or len(detected_faces) == 0:
                return [], "No faces detected in image. Try a clearer photo with better lighting."
            
            recognized_persons = []
            unrecognized_faces = 0
            debug_info = [f"Using {backend_used} detector - Found {len(detected_faces)} face(s)\n"]
            
            # Process EVERY detected face
            for face_idx, detected_face in enumerate(detected_faces):
                debug_info.append(f"--- Face {face_idx+1} ---")
                
                # Save detected face temporarily
                temp_face_path = f"temp_face_{face_idx}.jpg"
                try:
                    face_img = detected_face['face']
                    face_img = (face_img * 255).astype(np.uint8)
                    face_img = cv2.resize(face_img, (224, 224))
                    cv2.imwrite(temp_face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    
                    best_match = None
                    best_distance = float('inf')
                    best_match_name = None
                    
                    # Compare THIS face with EVERY registered person
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
                            
                            # Track best match for this face
                            if distance < best_distance:
                                best_distance = distance
                                best_match_name = name
                                if distance < 0.7:  # Only accept if below threshold
                                    best_match = (person_id, name)
                                    
                        except Exception as e:
                            debug_info.append(f"  {name}: Error - {str(e)}")
                            continue
                    
                    # Report result for this face
                    if best_match:
                        recognized_persons.append(best_match)
                        debug_info.append(f"  ✓ MATCH: {best_match[1]} ({best_distance:.3f})")
                    else:
                        unrecognized_faces += 1
                        if best_match_name:
                            debug_info.append(f"  ✗ Closest: {best_match_name} ({best_distance:.3f} - too high)")
                        else:
                            debug_info.append(f"  ✗ No match")
                    
                    debug_info.append("")  # Blank line
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_face_path):
                        os.remove(temp_face_path)
            
            # Remove duplicate matches (same person detected multiple times)
            recognized_persons = list(set(recognized_persons))
            
            # Always return debug info
            full_debug = "\n".join(debug_info)
            full_debug += f"\nSummary: {len(detected_faces)} face(s) detected, {len(recognized_persons)} recognized, {unrecognized_faces} unrecognized"
            
            if recognized_persons:
                return recognized_persons, full_debug
            else:
                return [], full_debug
            
        except Exception as e:
            return [], f"Error: {str(e)}"
    
    def mark_attendance(self, person_ids):
        """Mark attendance for multiple persons"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = str(date.today())
        current_time = datetime.now().strftime("%H:%M:%S")
        
        marked = []
        already_marked = []
        
        for person_id in person_ids:
            # Check if already marked today
            cursor.execute('''
                SELECT COUNT(*) FROM attendance 
                WHERE person_id = ? AND date = ?
            ''', (person_id, today))
            
            count = cursor.fetchone()[0]
            
            # Get person name
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
        """Generate attendance report"""
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
        """Get list of all enrolled persons"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT name, enrollment_date FROM persons')
        persons = cursor.fetchall()
        conn.close()
        return persons
    
    def delete_person(self, name):
        """Delete a person from the database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Get face image path
            cursor.execute('SELECT id, face_image_path FROM persons WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if not result:
                return False, "Person not found"
            
            person_id, face_path = result
            
            # Delete attendance records first (foreign key constraint)
            cursor.execute('DELETE FROM attendance WHERE person_id = ?', (person_id,))
            
            # Delete person
            cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
            
            # Commit before deleting file
            conn.commit()
            
            # Delete face image file
            if face_path and os.path.exists(face_path):
                try:
                    os.remove(face_path)
                except:
                    pass  # File deletion failure shouldn't fail the operation
            
            return True, "Person deleted successfully"
                
        except Exception as e:
            if conn:
                conn.rollback()
            return False, f"Error: {str(e)}"
        finally:
            if conn:
                conn.close()


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("900x700")
        
        self.system = AttendanceSystem()
        self.create_widgets()
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.enroll_tab = ttk.Frame(self.notebook)
        self.attendance_tab = ttk.Frame(self.notebook)
        self.report_tab = ttk.Frame(self.notebook)
        self.view_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.enroll_tab, text="Register Face")
        self.notebook.add(self.attendance_tab, text="Mark Attendance")
        self.notebook.add(self.report_tab, text="Reports")
        self.notebook.add(self.view_tab, text="Registered People")
        
        self.create_enroll_tab()
        self.create_attendance_tab()
        self.create_report_tab()
        self.create_view_tab()
    
    def create_enroll_tab(self):
        frame = ttk.LabelFrame(self.enroll_tab, text="Register New Person", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(frame, text="Name:", font=('Arial', 11)).grid(row=0, column=0, sticky='w', pady=10)
        self.name_entry = ttk.Entry(frame, width=35, font=('Arial', 11))
        self.name_entry.grid(row=0, column=1, pady=10, padx=10)
        
        ttk.Label(frame, text="Face Photo:", font=('Arial', 11)).grid(row=1, column=0, sticky='w', pady=10)
        self.image_path_label = ttk.Label(frame, text="No image selected", foreground='gray')
        self.image_path_label.grid(row=1, column=1, sticky='w', pady=10, padx=10)
        
        ttk.Button(frame, text="Select Photo", command=self.select_image, width=20).grid(
            row=2, column=1, pady=10, sticky='w', padx=10
        )
        
        ttk.Button(frame, text="Register Person", command=self.enroll_person, width=20).grid(
            row=3, column=0, columnspan=2, pady=20
        )
        
        ttk.Label(frame, text="Tip: Use a clear, front-facing photo with good lighting", 
                 foreground='gray', font=('Arial', 9)).grid(row=4, column=0, columnspan=2, pady=10)
        
        self.selected_image_path = None
    
    def create_attendance_tab(self):
        frame = ttk.LabelFrame(self.attendance_tab, text="Mark Attendance from Photo", padding=20)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(frame, text="Upload a photo (can be a group photo):", 
                 font=('Arial', 12, 'bold')).pack(pady=15)
        
        ttk.Button(frame, text="📷 Select Photo & Recognize Faces", 
                  command=self.mark_attendance_from_image,
                  width=35).pack(pady=10)
        
        self.attendance_status = tk.Text(frame, height=15, width=70, font=('Arial', 10), wrap='word')
        self.attendance_status.pack(pady=20, padx=10)
        
        scrollbar = ttk.Scrollbar(frame, command=self.attendance_status.yview)
        self.attendance_status.config(yscrollcommand=scrollbar.set)
        
        ttk.Label(frame, text="Note: Recognition may take 10-20 seconds for group photos", 
                 foreground='gray', font=('Arial', 9)).pack(pady=5)
    
    def create_report_tab(self):
        frame = ttk.Frame(self.report_tab, padding=20)
        frame.pack(fill='both', expand=True)
        
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
        ttk.Button(date_frame, text="Show All Records", 
                  command=self.show_all_records).pack(side='left', padx=5)
        ttk.Button(date_frame, text="Export to CSV", 
                  command=self.export_report).pack(side='left', padx=5)
        
        columns = ('Name', 'Date', 'Time')
        self.report_tree = ttk.Treeview(frame, columns=columns, show='headings', height=22)
        
        for col in columns:
            self.report_tree.heading(col, text=col)
            self.report_tree.column(col, width=250)
        
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.report_tree.yview)
        self.report_tree.configure(yscrollcommand=scrollbar.set)
        
        self.report_tree.pack(side='left', fill='both', expand=True, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
        
        # Auto-load all records on startup
        self.root.after(100, self.show_all_records)
    
    def create_view_tab(self):
        frame = ttk.Frame(self.view_tab, padding=20)
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Registered People", font=('Arial', 14, 'bold')).pack(pady=10)
        
        columns = ('Name', 'Enrollment Date')
        self.persons_tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.persons_tree.heading(col, text=col)
            self.persons_tree.column(col, width=300)
        
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.persons_tree.yview)
        self.persons_tree.configure(yscrollcommand=scrollbar.set)
        
        self.persons_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_persons_list).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Delete Selected", command=self.delete_selected_person).pack(side='left', padx=5)
        
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
        
        if not name or not self.selected_image_path:
            messagebox.showerror("Error", "Please enter a name and select a photo")
            return
        
        self.root.config(cursor="wait")
        self.root.update()
        
        success, message = self.system.enroll_person(name, self.selected_image_path)
        
        self.root.config(cursor="")
        
        if success:
            messagebox.showinfo("Success", message)
            self.name_entry.delete(0, 'end')
            self.image_path_label.config(text="No image selected", foreground='gray')
            self.selected_image_path = None
            self.refresh_persons_list()
        else:
            messagebox.showerror("Error", message)
    
    def mark_attendance_from_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Photo (Single Person or Group)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        self.attendance_status.delete(1.0, tk.END)
        self.attendance_status.insert(tk.END, "🔍 Analyzing photo and recognizing faces...\n")
        self.attendance_status.insert(tk.END, "This may take 10-20 seconds for group photos...\n\n")
        self.root.config(cursor="wait")
        self.root.update()
        
        recognized_persons, message = self.system.recognize_faces_in_image(file_path)
        
        self.root.config(cursor="")
        
        # Always show debug info first
        self.attendance_status.delete(1.0, tk.END)
        self.attendance_status.insert(tk.END, "=== DETECTION & RECOGNITION DETAILS ===\n\n", 'header')
        self.attendance_status.insert(tk.END, message + "\n\n")
        
        if not recognized_persons:
            self.attendance_status.insert(tk.END, "=" * 50 + "\n\n", 'header')
            self.attendance_status.insert(tk.END, "❌ No recognized faces\n", 'error')
            self.attendance_status.tag_config('header', font=('Arial', 10, 'bold'))
            self.attendance_status.tag_config('error', foreground='red')
            return
        
        # Mark attendance for all recognized persons
        marked, already_marked = self.system.mark_attendance([p[0] for p in recognized_persons])
        
        self.attendance_status.insert(tk.END, "=" * 50 + "\n\n", 'header')
        self.attendance_status.insert(tk.END, "✅ RECOGNITION COMPLETE\n\n", 'success')
        
        self.attendance_status.insert(tk.END, f"Recognized {len(recognized_persons)} person(s):\n")
        for _, name in recognized_persons:
            self.attendance_status.insert(tk.END, f"  • {name}\n")
        self.attendance_status.insert(tk.END, "\n")
        
        if marked:
            self.attendance_status.insert(tk.END, "✓ Attendance Marked:\n", 'success')
            for name in marked:
                self.attendance_status.insert(tk.END, f"  • {name}\n")
            self.attendance_status.insert(tk.END, "\n")
        
        if already_marked:
            self.attendance_status.insert(tk.END, "⚠ Already marked today:\n", 'warning')
            for name in already_marked:
                self.attendance_status.insert(tk.END, f"  • {name}\n")
        
        self.attendance_status.tag_config('header', font=('Arial', 10, 'bold'))
        self.attendance_status.tag_config('success', foreground='green')
        self.attendance_status.tag_config('warning', foreground='orange')
        self.attendance_status.tag_config('error', foreground='red')
    
    def generate_report(self):
        start = self.start_date_entry.get().strip()
        end = self.end_date_entry.get().strip()
        
        if not start or not end:
            messagebox.showwarning("Warning", "Please enter both start and end dates")
            return
        
        df = self.system.get_attendance_report(start, end)
        
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        if df.empty:
            messagebox.showinfo("No Records", f"No attendance records found between {start} and {end}")
        else:
            for _, row in df.iterrows():
                self.report_tree.insert('', 'end', values=tuple(row))
        
        self.current_report_df = df
    
    def show_all_records(self):
        """Show all attendance records without date filter"""
        df = self.system.get_attendance_report(None, None)
        
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        if df.empty:
            # Don't show messagebox on auto-load, just leave empty
            pass
        else:
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
        for item in self.persons_tree.get_children():
            self.persons_tree.delete(item)
        
        persons = self.system.get_all_persons()
        for person in persons:
            self.persons_tree.insert('', 'end', values=person)
    
    def delete_selected_person(self):
        selected = self.persons_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
        
        item = self.persons_tree.item(selected[0])
        name = item['values'][0]
        
        if messagebox.askyesno("Confirm Delete", f"Delete {name} and all their attendance records?"):
            success, message = self.system.delete_person(name)
            if success:
                messagebox.showinfo("Success", message)
                self.refresh_persons_list()
            else:
                messagebox.showerror("Error", message)


def main():
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()