import requests
import os

# API URL
url = 'http://localhost:5000/dance_videos'

# Create dummy video files for testing
def create_dummy_file(filename):
    with open(filename, 'wb') as f:
        f.write(b'fake video content')
    return filename

print("Creating dummy files...")
teacher_file = create_dummy_file('test_teacher.mp4')
student_file = create_dummy_file('test_student.mp4')

files = {
    'teacher': open(teacher_file, 'rb'),
    'student': open(student_file, 'rb')
}

try:
    print(f"Sending POST request to {url}...")
    response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {response.json()}")
    except:
        print(f"Response Text: {response.text}")
        
    if response.status_code == 200:
        print("\nSUCCESS: Files uploaded successfully.")
    else:
        print("\nFAILURE: Upload failed.")

except requests.exceptions.ConnectionError:
    print(f"\nERROR: Could not connect to {url}. Is fileGetter.py running?")
except Exception as e:
    print(f"\nERROR: {e}")

finally:
    # Close file handles and cleanup
    files['teacher'].close()
    files['student'].close()
    
    if os.path.exists(teacher_file):
        os.remove(teacher_file)
    if os.path.exists(student_file):
        os.remove(student_file)
    print("Cleanup done.")
