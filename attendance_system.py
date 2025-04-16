import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import csv
from datetime import datetime
import face_recognition

# Define the name of the directory to be created
directory = "dataset"

# Check if directory already exists
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to capture student images
def capture_images():
    student_name = name_entry.get()
    roll_number = roll_entry.get()

    if student_name and roll_number:
        student_folder = os.path.join(directory, student_name)

        # Check if the student's folder already exists
        if os.path.exists(student_folder):
            messagebox.showwarning("User Exists", f"Images for {student_name} already exist.")
            return
        
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Capture Image")
        img_counter = 0

        # Create a new folder for the student
        os.makedirs(student_folder)

        while True:
            ret, frame = cam.read()
            if not ret:
                break
            cv2.imshow("Capture Image", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC pressed
                break
            elif k % 256 == 32:  # SPACE pressed
                img_name = f"{student_name}_{roll_number}_{img_counter}.png"
                cv2.imwrite(os.path.join(student_folder, img_name), frame)
                img_counter += 1

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", "Images captured successfully!")
    else:
        messagebox.showwarning("Input Error", "Please enter both name and roll number.")

# Function to delete student images
def delete_student_images():
    student_name = name_entry.get()

    if student_name:
        student_folder = os.path.join(directory, student_name)
        if os.path.exists(student_folder):
            for file in os.listdir(student_folder):
                os.remove(os.path.join(student_folder, file))
            os.rmdir(student_folder)
            messagebox.showinfo("Success", f"Deleted all images for {student_name}.")
        else:
            messagebox.showwarning("Not Found", f"No images found for {student_name}.")
    else:
        messagebox.showwarning("Input Error", "Please enter the student name.")

# Function to clear attendance records
def clear_attendance_records():
    if os.path.exists('attendance.csv'):
        os.remove('attendance.csv')
        messagebox.showinfo("Success", "Attendance records cleared.")
    else:
        messagebox.showwarning("Not Found", "No attendance records found.")

# Function to mark attendance
def mark_attendance():
    known_face_encodings = []
    known_face_names = []
    known_roll_numbers = []  # List to hold roll numbers

    # Load known faces and their encodings
    for student_folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, student_folder)):
            for img_file in os.listdir(os.path.join(directory, student_folder)):
                img_path = os.path.join(directory, student_folder, img_file)
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:  # Check if encoding is found
                    known_face_encodings.append(encoding[0])
                    # Extract roll number from the filename
                    roll_number = img_file.split("_")[1]  # Assuming the format is name_rollnumber_index.png
                    known_face_names.append(student_folder)
                    known_roll_numbers.append(roll_number)  # Save the corresponding roll number

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        messagebox.showerror("Camera Error", "Could not access the webcam. Please check your camera settings.")
        return

    cv2.namedWindow("Mark Attendance")
    attendance_recorded = False  # Flag to track if attendance was marked
    recognized_any = False  # Flag to track if any faces were recognized

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            roll_number = None

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                roll_number = known_roll_numbers[best_match_index]  # Get the corresponding roll number
                recognized_any = True  # Set flag to True since a face was recognized

                # Mark attendance
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open('attendance.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, roll_number, 'Present', timestamp])
                attendance_recorded = True  # Set flag to True
                
                # Draw rectangle around face and display name
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} - Present", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # If no match, set the flag and display a message
                recognized_any = True  # Set this to True to indicate we processed a face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown - Not Present", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the original frame with annotations
        cv2.imshow("Mark Attendance", frame)

        if cv2.waitKey(1) % 256 == 27:  # ESC pressed
            break

    cam.release()
    cv2.destroyAllWindows()

    # Show message after attendance marking
    if attendance_recorded:
        messagebox.showinfo("Attendance", "Attendance marked successfully!")
    elif not recognized_any:
        messagebox.showwarning("Attendance", "No recognized faces found. Attendance not marked.")
    else:
        messagebox.showwarning("Attendance", "Face not recognized. Attendance not marked.")

# Set up the GUI
app = tk.Tk()
app.title("Student Attendance System")
app.geometry("800x600")  # Set the window size to be larger
app.config(bg='pink')  # Change the background color

# Styling the buttons and labels
button_style = {
    "padx": 20,
    "pady": 10,
    "font": ('Arial', 23),
    "bg": "white",
    "fg": "black",
    "activebackground": "darkorange",
    "activeforeground": "black",
}

tk.Label(app, text="Student Name:", font=('Arial', 25), bg='white').grid(row=0, column=0, pady=30, padx=20)
tk.Label(app, text="Roll Number:", font=('Arial', 25), bg='white').grid(row=1, column=0, pady=30, padx=10)

name_entry = tk.Entry(app, font=('Arial', 20))
roll_entry = tk.Entry(app, font=('Arial', 20))

name_entry.grid(row=0, column=1, padx=20, pady=30)
roll_entry.grid(row=1, column=1, padx=20, pady=30)

capture_button = tk.Button(app, text="Capture Images", command=capture_images, **button_style)
capture_button.grid(row=4, column=0, columnspan=2, pady=40)

delete_button = tk.Button(app, text="Delete Images", command=delete_student_images, **button_style)
delete_button.grid(row=4, column=3, columnspan=2, pady=40)

clear_attendance_button = tk.Button(app, text="Clear Attendance", command=clear_attendance_records, **button_style)
clear_attendance_button.grid(row=6 ,column=0, columnspan=2, pady=40)

attendance_button = tk.Button(app, text="Mark Attendance", command=mark_attendance, **button_style)
attendance_button.grid(row=6, column=3, columnspan=2, pady=40)

app.mainloop()
