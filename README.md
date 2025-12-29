# AI-Powered-Attendance-Tracker

# Face Recognition Attendance Web App

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install flask opencv-python deepface tf-keras pandas pillow
```

### 2. Project Structure

Create this folder structure:

```
attendance-web-app/
├── app.py
├── templates/
│   └── index.html
├── uploads/          (created automatically)
└── registered_faces/ (created automatically)
```

### 3. Run the Application

```bash
python app.py
```

The app will start at: **http://localhost:5000**

### 4. Access from Other Devices

To access from phones/tablets on the same network:

1. Find your computer's IP address:
   - Windows: `ipconfig` (look for IPv4)
   - Mac/Linux: `ifconfig` or `ip addr`
   
2. Access from other devices: `http://YOUR_IP:5000`
   - Example: `http://192.168.1.100:5000`

## 📱 Features

### ✅ Registration
- Use webcam or upload photo
- Register new people with face recognition
- Preview before registration

### ✅ Attendance Marking
- Capture group photos
- Automatic face detection and recognition
- Detailed debug information
- Works with webcam or uploaded photos

### ✅ Reports
- View all attendance records
- Filter by date range
- Export to CSV
- Real-time data updates

### ✅ People Management
- View all registered people
- Delete registrations
- See enrollment dates

## 🌐 Deploy to Production

### Option 1: Heroku (Free Tier)

1. Create a `requirements.txt`:
```
flask
opencv-python-headless
deepface
tf-keras
pandas
pillow
gunicorn
```

2. Create `Procfile`:
```
web: gunicorn app:app
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 2: DigitalOcean / AWS / Azure

1. Set up a server
2. Install dependencies
3. Use nginx + gunicorn
4. Configure SSL certificate
5. Set up domain name

### Option 3: PythonAnywhere (Easiest)

1. Sign up at pythonanywhere.com
2. Upload your code
3. Configure web app
4. Your app will be at: `yourusername.pythonanywhere.com`

## 🔒 Security Considerations

For production deployment:

1. **Change the secret key** in `app.py`:
```python
app.config['SECRET_KEY'] = 'your-secure-random-key-here'
```

2. **Add authentication**:
   - User login/signup
   - Admin dashboard
   - Role-based access

3. **Add HTTPS**:
   - Use Let's Encrypt SSL certificate
   - Force HTTPS redirect

4. **Database upgrade**:
   - Use PostgreSQL instead of SQLite
   - Better for concurrent users

5. **Rate limiting**:
   - Prevent abuse
   - Limit API calls

## 💡 Tips

- **Good lighting** improves recognition accuracy
- **Clear photos** work best for enrollment
- **Front-facing** photos are most reliable
- **Group photos** can recognize multiple people at once

## 🐛 Troubleshooting

**Camera not working?**
- Check browser permissions
- Try different browser (Chrome recommended)
- Ensure HTTPS for production (required for camera)

**Recognition not accurate?**
- Ensure good lighting during enrollment
- Use clear, front-facing photos
- Re-register with better quality photos

**Slow processing?**
- First run downloads AI models (one-time, ~500MB)
- Processing takes 10-20 seconds for group photos
- Consider upgrading server specs for production

## 📊 Next Steps

To make it even better:

1. **Add email notifications** for absences
2. **Create analytics dashboard** with charts
3. **Add mobile app** (React Native)
4. **Real-time video stream** recognition
5. **Integration with existing systems** (via API)
6. **Multi-organization support**
7. **Backup and export** features

## 🎯 Resume Project Description

**Face Recognition Attendance System**
- Developed full-stack web application using Flask, Python, and DeepFace AI
- Implemented real-time face detection and recognition using Facenet512 model
- Built responsive web interface with HTML/CSS/JavaScript and webcam integration
- Created RESTful API for attendance management with SQLite database
- Supports multi-face detection in group photos with 70% accuracy threshold
- Features include user registration, attendance tracking, reporting, and CSV export
- Deployed with Docker containerization for easy scaling

**Technologies:** Python, Flask, DeepFace, OpenCV, TensorFlow, SQLite, HTML/CSS/JavaScript, REST API