import cv2  # OpenCV - for video reading and image processing
import mediapipe as mp  # Google's MediaPipe - for face detection
import os  # For file/folder operations
from tqdm import tqdm  # Progress bar library (shows processing progress)


def extract_faces_from_video(video_path, output_folder, video_name):
    """Extract faces from a single video and return count

        Process:
        1. Open video with OpenCV
        2. Sample frames at 5 FPS (every 6th frame for 30 FPS video)
        3. Detect faces in each sampled frame using MediaPipe
        4. Crop face with 30% margin
        5. Resize to 224x224
        6. Save as JPEG
    """
    # Get the face detection module from MediaPipe solutions
    mp_face_detection = mp.solutions.face_detection

    # Creating the face deteciton object  
    face_detection = mp_face_detection.FaceDetection(

        # Setting the model to model 0 which works better for faces within 2 meters
        model_selection=0,

        # setting a confidence threshold
        min_detection_confidence=0.5
    )
    
    # Open the file with openCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return 0  
    
    # Calcuating the frame sampling rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    target_fps = 5
    frame_skip = int(fps / target_fps) if fps > 0 else 1
    

    # Creating the output folder 
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    saved_count = 0

    # looping as long as the video is open    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            height, width, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * width)
                y = int(bboxC.ymin * height)
                w = int(bboxC.width * width)
                h = int(bboxC.height * height)
                
                margin = 0.3
                x_margin = int(w * margin)
                y_margin = int(h * margin)
                
                x1 = max(0, x - x_margin)
                y1 = max(0, y - y_margin)
                x2 = min(width, x + w + x_margin)
                y2 = min(height, y + h + y_margin)
                
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:  
                    face_resized = cv2.resize(face_crop, (224, 224))
                    
                    video_id = os.path.splitext(video_name)[0]  
                    output_path = f"{output_folder}/{video_id}_frame_{frame_count}.jpg"
                    cv2.imwrite(output_path, face_resized)
                    saved_count += 1
        
        frame_count += 1
    
    cap.release()
    face_detection.close()
    
    return saved_count  

# Process all the videos in a folder and extract the faces
def process_folder(video_folder, output_folder, label):
    """Process all videos in a folder"""

    # Get a list of all the mp4 video files 
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    # Print processing header
    print(f"\n{'='*50}")
    print(f"Processing {label} videos")
    print(f"Found {len(video_files)} videos")
    print(f"{'='*50}\n")
    
    total_faces = 0
    
    # This is the code for the progress bar
    for video_file in tqdm(video_files, desc=f"Processing {label}"):
        video_path = os.path.join(video_folder, video_file)
        faces_saved = extract_faces_from_video(video_path, output_folder, video_file)
        total_faces += faces_saved
    
    print(f"\n{label} complete: {total_faces} faces extracted\n")
    return total_faces

if __name__ == "__main__":
    print("\n" + "="*50)
    print("PROCESSING ALL VIDEOS")
    print("="*50)
    print(f"Estimated time: 1.2 hours")
    print(f"Expected faces: ~450,000")
    print("="*50 + "\n")
    
    real_video_folder = "data/videos/Celeb-DF-v2/Celeb-real"
    real_output_folder = "data/faces/real"
    real_faces = process_folder(real_video_folder, real_output_folder, "REAL")
    
    fake_video_folder = "data/videos/Celeb-DF-v2/Celeb-synthesis"
    fake_output_folder = "data/faces/fake"
    fake_faces = process_folder(fake_video_folder, fake_output_folder, "FAKE")
    
    print(f"\n{'='*60}")
    print(f" EXTRACTION COMPLETE! ")
    print(f"{'='*60}")
    print(f"Real faces: {real_faces:,}")
    print(f"Fake faces: {fake_faces:,}")
    print(f"Total faces: {real_faces + fake_faces:,}")
    print(f"{'='*60}\n")