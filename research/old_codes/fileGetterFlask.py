import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure upload folder
# Use absolute path relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root, then into uploaded_videos
UPLOAD_FOLDER = os.path.join(os.path.dirname(CURRENT_DIR), 'uploaded_videos')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/dance_videos', methods=['POST'])
def upload_files():
    if 'teacher' not in request.files or 'student' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    teacher_file = request.files['teacher']
    student_file = request.files['student']
    
    if teacher_file.filename == '' or student_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Secure filenames could be added here, but sticking to simple requirement first
    teacher_path = os.path.join(UPLOAD_FOLDER, teacher_file.filename)
    student_path = os.path.join(UPLOAD_FOLDER, student_file.filename)
    
    teacher_file.save(teacher_path)
    student_file.save(student_path)
    
    print(f"Saved files to: {teacher_path} and {student_path}")
    
    return jsonify({
        'message': 'Files uploaded successfully', 
        'teacher_path': teacher_path, 
        'student_path': student_path
    }), 200

if __name__ == '__main__':
    print(f"Starting server on http://localhost:5000/dance_videos")
    app.run(debug=True, port=5000)
