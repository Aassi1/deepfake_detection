import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from torchvision import transforms
import mediapipe as mp
from datetime import datetime
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from model import get_model

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>

    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    
    .verdict-fake {
        background: #2c3e50;  
        border-left: 4px solid #e74c3c;  
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    
    .verdict-real {
        background: #2c3e50;  
        border-left: 4px solid #3498db;  
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .verdict-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #ffffff;  
    }
    
    .verdict-confidence {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #ffffff;  
    }
    
    .stat-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3498db;
        margin: 1rem 0;
        color: #333;
    }
    
    
    .sidebar-metric {
        background: #2c3e50;  
        color: #ffffff;  
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border: 1px solid #34495e;
    }
    
    .sidebar-metric strong {
        color: #3498db;  
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device, pretrained=False, freeze_backbone=True)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def log_feedback(video_name, predicted_label, true_label, is_correct, num_faces):
    """Log user feedback"""
    feedback_file = 'feedback_log.csv'
    
    feedback_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_name': video_name,
        'predicted_label': predicted_label,
        'true_label': true_label,
        'is_correct': is_correct,
        'num_faces': num_faces
    }
    
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
    else:
        df = pd.DataFrame([feedback_data])
    
    df.to_csv(feedback_file, index=False)

def get_feedback_stats():
    feedback_file = 'feedback_log.csv'
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        total = len(df)
        correct = len(df[df['is_correct'] == True])
        return {'total': total, 'correct': correct, 'accuracy': (correct/total*100) if total > 0 else 0}
    return {'total': 0, 'correct': 0, 'accuracy': 0}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_faces_from_video(video_path, max_frames=30):
    """Extract faces from video"""
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / 5))
    
    faces = []
    frame_count = 0
    frames_processed = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened() and frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            status_text.text(f"Extracting faces: {frames_processed + 1}/{max_frames}")
            progress_bar.progress((frames_processed + 1) / max_frames)
            
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
                x1 = max(0, int(x - w * margin))
                y1 = max(0, int(y - h * margin))
                x2 = min(width, int(x + w * (1 + margin)))
                y2 = min(height, int(y + h * (1 + margin)))
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    faces.append(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    frames_processed += 1
        
        frame_count += 1
    
    progress_bar.empty()
    status_text.empty()
    cap.release()
    face_detection.close()
    return faces

def classify_face(face_img, model, device):
    pil_img = Image.fromarray(face_img)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence


def main():
    st.markdown('<h1 class="main-header">Deepfake Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered video authenticity verification</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Model Stats")
        
        with st.spinner("Loading model..."):
            model, device = load_model()
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <strong>Device:</strong> {str(device).upper()}
        </div>
        <div class="sidebar-metric">
            <strong>Test Accuracy:</strong> 88.84%
        </div>
        <div class="sidebar-metric">
            <strong>Precision (Fake):</strong> 96.73%
        </div>
        <div class="sidebar-metric">
            <strong>ROC-AUC:</strong> 0.91
        </div>
        """, unsafe_allow_html=True)
        
        feedback_stats = get_feedback_stats()
        if feedback_stats['total'] > 0:
            st.markdown("---")
            st.markdown("### User Feedback")
            st.markdown(f"""
            <div class="sidebar-metric">
                <strong>Predictions:</strong> {feedback_stats['total']}<br>
                <strong>Accuracy:</strong> {feedback_stats['accuracy']:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
    # Info
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Upload a video, and the system will extract faces, 
        analyze each frame using a CNN, and determine if the video is authentic or a deepfake.
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Save temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        video_name = uploaded_file.name
        
        # Show video info
        st.markdown("**Video Information:**")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("FPS", f"{fps:.0f}")
        with col3:
            st.metric("Frames", f"{frame_count}")
        with col4:
            st.metric("Resolution", f"{width}×{height}")

        st.markdown("---")
        
        # Analyze button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            
            # Extract faces
            with st.spinner("Extracting faces..."):
                faces = extract_faces_from_video(video_path, max_frames=30)
            
            if len(faces) == 0:
                st.error("No faces detected. Please try another video.")
                try:
                    os.unlink(video_path)
                except:
                    pass
                return
            
            # Classify
            with st.spinner(f"Analyzing {len(faces)} faces..."):
                predictions = []
                for face in faces:
                    pred, conf = classify_face(face, model, device)
                    predictions.append((pred, conf))
            
            # Calculate results
            fake_count = sum(1 for p, _ in predictions if p == 1)
            real_count = len(predictions) - fake_count
            
            fake_confs = [c for p, c in predictions if p == 1]
            real_confs = [c for p, c in predictions if p == 0]
            
            avg_fake_conf = np.mean(fake_confs) if fake_count > 0 else 0
            avg_real_conf = np.mean(real_confs) if real_count > 0 else 0
            
            overall_verdict = "FAKE" if fake_count > real_count else "REAL"
            verdict_confidence = max((fake_count / len(predictions)) * 100, 
                                    (real_count / len(predictions)) * 100)
            
            st.markdown("---")
            st.markdown("## Analysis Results")
            
            # Verdict box
            if overall_verdict == "FAKE":
                st.markdown(f"""
                <div class="verdict-fake">
                    <div class="verdict-title"> Deepfake Detected</div>
                    <div class="verdict-confidence">{verdict_confidence:.1f}%</div>
                    <div style="color: #666;">Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-real">
                    <div class="verdict-title">✓ Appears Authentic</div>
                    <div class="verdict-confidence">{verdict_confidence:.1f}%</div>
                    <div style="color: #666;">Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{len(faces)}</div>
                    <div class="stat-label">Frames Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{real_count}</div>
                    <div class="stat-label">Real ({avg_real_conf*100:.1f}% avg)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{fake_count}</div>
                    <div class="stat-label">Fake ({avg_fake_conf*100:.1f}% avg)</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sample faces
            st.markdown("### Detected Faces")
            
            num_display = min(10, len(faces))
            cols = st.columns(5)
            
            for idx in range(num_display):
                face = faces[idx]
                pred, conf = predictions[idx]
                label = "FAKE" if pred == 1 else "REAL"
                emoji = "🔴" if pred == 1 else "🟢"
                
                with cols[idx % 5]:
                    st.image(face, caption=f"{emoji} {label} ({conf*100:.0f}%)", width=150)
            
            # Feedback
            st.markdown("---")
            st.markdown("### Feedback")
            st.markdown("Was this prediction correct?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✓ Correct", use_container_width=True):
                    log_feedback(video_name, overall_verdict, overall_verdict, True, len(faces))
                    st.success("Feedback recorded!")
            
            with col2:
                if st.button("✗ Actually Real", use_container_width=True):
                    log_feedback(video_name, overall_verdict, "REAL", False, len(faces))
                    st.success("Correction recorded!")
            
            with col3:
                if st.button("✗ Actually Fake", use_container_width=True):
                    log_feedback(video_name, overall_verdict, "FAKE", False, len(faces))
                    st.success("Correction recorded!")
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass

if __name__ == "__main__":
    main()