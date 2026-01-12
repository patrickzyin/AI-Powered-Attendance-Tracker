# AI-Powered Face Recognition Attendance System

A full-stack web application that automates attendance tracking using deep learning facial recognition. Built with Python, Flask, DeepFace, and modern web technologies.

## Technical Stack

**Backend:** Python, Flask, DeepFace, TensorFlow, OpenCV, SQLite, Pandas  
**Frontend:** JavaScript (ES6+), HTML5, CSS3  
**AI/ML:** Facenet512 (99.63% accuracy), RetinaFace, MTCNN

## Key Features

### User Authentication
- Secure registration and login system
- Password hashing with bcrypt
- Session-based authentication
- Multi-tenant architecture (complete data isolation between users)

### Face Enrollment
- Register people with webcam capture or photo upload
- Single-face validation during enrollment
- Automatic face detection and normalization
- Real-time preview before saving

### Attendance Marking
- Group photo recognition (mark multiple people at once)
- Multi-backend face detection for reliability
- Automatic duplicate handling (updates time instead of creating duplicates)
- Supports both webcam and uploaded images

### Reporting & Management
- View attendance records with date filtering
- Export to CSV for analysis
- Manage registered people (view and delete)
- Complete audit trail with timestamps

## Installation

### Prerequisites
- Python 3.8+
- Webcam (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/patrickzyin/AI-Powered-Attendance-Tracker.git
cd ai-attendance-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask opencv-python deepface pandas werkzeug tensorflow

# Run application
python app.py
```

Access at: http://localhost:5000

## Usage Guide

### 1. Create Account
- Navigate to http://localhost:5000
- Click "Get Started" or "Register"
- Fill in username, email, password, and organization (optional)
- Password requirements: 8+ characters, 1 uppercase, 1 number

### 2. Enroll People
- Go to "Register" tab
- Enter person's full name
- Capture photo (webcam) or upload image
- Important: Only one face per enrollment photo
- Click "Register Person"

### 3. Mark Attendance
- Go to "Attendance" tab
- Capture or upload photo (can include multiple people)
- System identifies all registered faces
- Results show:
  - Green checkmark: New attendance marked
  - Blue refresh icon: Updated attendance (replaces earlier time)

### 4. View Reports
- Go to "Reports" tab
- Click "Show All" for complete history
- Or filter by date range
- Click "Export CSV" to download data

### 5. Manage People
- Go to "People" tab
- View all registered individuals
- Delete people as needed (removes all their attendance records)

## Project Structure

```
ai-attendance-tracker/
├── app.py                    # Main Flask application
├── templates/
│   ├── landing.html          # Landing page
│   ├── login.html            # Login/register page
│   └── dashboard.html        # Main dashboard
├── uploads/                  # Temporary image storage
├── registered_faces/         # Face embeddings (organized by user)
└── attendance.db            # SQLite database
```


## License

MIT License

## Author

**Patrick Yin**  
GitHub: [@patrickzyin](https://github.com/patrickzyin)

---

**Built with Python, Flask, DeepFace, and Computer Vision**
