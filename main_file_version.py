import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

def process_video_for_blink_rate(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    blink_count = 0
    processed_frames = 0
    face_detected_count = 0
    blink_active = False
    
    sample_frame = None
    frame_skip = 2
    
    base_options = mp.tasks.BaseOptions(model_asset_path="face_landmarker.task")
    
    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_face_blendshapes=True
    )
    
    print(f"Starting blink detection on {video_path.name} (using blendshapes)")
    print(f"  Frames: {total_frames} | Duration: ~{total_frames/fps:.1f} sec")
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        last_print_time = time.time()
        early_print_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            
            if (processed_frames - 1) % frame_skip != 0:
                continue
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(processed_frames * (1000 / fps))
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.face_landmarks and len(result.face_landmarks) > 0 and result.face_blendshapes:
                face_detected_count += 1
                blendshapes = result.face_blendshapes[0]
                
                eye_blink_left = next((bs.score for bs in blendshapes if bs.category_name == "eyeBlinkLeft"), 0.0)
                eye_blink_right = next((bs.score for bs in blendshapes if bs.category_name == "eyeBlinkRight"), 0.0)
                avg_blink_score = (eye_blink_left + eye_blink_right) / 2.0
                
                if avg_blink_score > 0.6 and not blink_active:
                    blink_active = True
                elif avg_blink_score < 0.3 and blink_active:
                    blink_count += 1
                    blink_active = False
                
                if sample_frame is None and face_detected_count > 30:
                    sample_frame = frame.copy()
                    annotated = frame.copy()
                    for lm in result.face_landmarks[0]:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
                    sample_frame = annotated
            
            if time.time() - last_print_time > (6 if early_print_count < 4 else 10):
                progress = (processed_frames / total_frames) * 100
                print(f"  Progress: {progress:.1f}%  |  Blinks so far: {blink_count}  |  Face frames: {face_detected_count}")
                last_print_time = time.time()
                early_print_count += 1
    
    cap.release()
    
    duration_seconds = total_frames / fps if fps > 0 else 1.0
    blink_rate_per_sec = blink_count / duration_seconds if duration_seconds > 0 else 0.0
    
    print(f"Finished {video_path.name} → {blink_count} blinks detected ({blink_rate_per_sec:.4f} blinks/sec | {blink_rate_per_sec*60:.1f} blinks/min)")
    
    return {
        'video': video_path.name,
        'total_blinks': blink_count,
        'duration_sec': round(duration_seconds, 2),
        'blink_rate_per_sec': round(blink_rate_per_sec, 4),
        'blink_rate_per_min': round(blink_rate_per_sec * 60, 2),
        'sample_frame': sample_frame
    }


def estimate_facial_dimensions(landmarks, frame_shape):
    h, w = frame_shape[:2]
    
    forehead_idx = 10
    chin_idx = 152
    left_ear_idx = 234
    right_ear_idx = 454
    left_eye_outer = 33
    left_eye_inner = 133
    nose_bridge = 168
    nose_tip = 4
    mouth_left = 61
    mouth_right = 291
    
    def pixel_dist(idx1, idx2):
        p1 = np.array([landmarks[idx1].x * w, landmarks[idx1].y * h])
        p2 = np.array([landmarks[idx2].x * w, landmarks[idx2].y * h])
        return np.linalg.norm(p1 - p2)
    
    face_height_px = pixel_dist(forehead_idx, chin_idx)
    face_width_px = pixel_dist(left_ear_idx, right_ear_idx)
    eye_width_px = pixel_dist(left_eye_outer, left_eye_inner)
    nose_length_px = pixel_dist(nose_bridge, nose_tip)
    mouth_width_px = pixel_dist(mouth_left, mouth_right)
    
    scale_cm_per_px = 14.5 / face_width_px if face_width_px > 50 else 0.1
    
    return {
        'face_height_cm': round(face_height_px * scale_cm_per_px, 1),
        'face_width_cm': round(face_width_px * scale_cm_per_px, 1),
        'eye_width_cm': round(eye_width_px * scale_cm_per_px, 1),
        'nose_length_cm': round(nose_length_px * scale_cm_per_px, 1),
        'mouth_width_cm': round(mouth_width_px * scale_cm_per_px, 1),
        'scale_cm_per_px': round(scale_cm_per_px, 4)
    }


def main():
    video_folder = Path("videos")
    video_files = sorted(list(video_folder.glob("*.mp4")))
    
    if not video_files:
        print("No .mp4 videos found in the 'videos' folder.")
        return
    
    print(f"Found {len(video_files)} video(s) in videos/ folder:")
    for i, v in enumerate(video_files, 1):
        print(f"  {i}. {v.name}")
    
    print("\nEnter your choice:")
    print("   - Enter number(s) separated by comma (e.g. 1,2)")
    print("   - Enter 'all' to process all videos")
    choice = input("Choice: ").strip().lower()
    
    if choice == "all":
        selected_videos = video_files
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected_videos = [video_files[i] for i in indices if 0 <= i < len(video_files)]
        except:
            print("Invalid input. Processing all videos.")
            selected_videos = video_files
    
    if not selected_videos:
        print("No valid videos selected.")
        return
    
    # Task A: Blink Rate
    print("TASK A: Blink Rate Analysis")
    
    results_a = []
    for video_path in selected_videos:
        print(f"\nProcessing {video_path.name} for blink rate...")
        result = process_video_for_blink_rate(video_path)
        results_a.append(result)
        
        if result['sample_frame'] is not None:
            plt.figure(figsize=(10, 6))
            rgb_sample = cv2.cvtColor(result['sample_frame'], cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_sample)
            plt.title(f"Sample frame from {result['video']} with landmarks")
            plt.axis('off')
            plt.show()
        
        print(f" Blinks detected: {result['total_blinks']}")
        print(f" Duration: {result['duration_sec']} seconds")
        print(f" Blink rate: {result['blink_rate_per_sec']:.4f} blinks/sec ({result['blink_rate_per_min']:.1f} blinks/min)")
    
    if results_a:
        overall_avg = np.mean([r['blink_rate_per_min'] for r in results_a])
        print(f"\nOverall average blink rate: {overall_avg:.1f} blinks per minute")
    
    # Task B: Facial Dimensions
    print("TASK B: Facial Dimensions")
    
    results_b = []
    for video_path in selected_videos:
        print(f"\nProcessing {video_path.name} for dimensions...")
        cap = cv2.VideoCapture(str(video_path))
        mid_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"  Could not read frame from {video_path.name}")
            continue
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        base_options = mp.tasks.BaseOptions(model_asset_path="face_landmarker.task")
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        
        with FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
            
            if result.face_landmarks and len(result.face_landmarks) > 0:
                landmarks = result.face_landmarks[0]
                dims = estimate_facial_dimensions(landmarks, frame.shape)
                
                results_b.append({
                    'video': video_path.name,
                    **dims
                })
                
                annotated = frame.copy()
                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
                
                plt.figure(figsize=(10, 7))
                plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                plt.title(f"Facial landmarks - {video_path.name}")
                plt.axis('off')
                plt.show()
                
                print(f"  Face height: {dims['face_height_cm']} cm")
                print(f"  Face width:  {dims['face_width_cm']} cm")
                print(f"  Eye width:   {dims['eye_width_cm']} cm")
                print(f"  Nose length: {dims['nose_length_cm']} cm")
                print(f"  Mouth width: {dims['mouth_width_cm']} cm")
            else:
                print(f"  No face detected in middle frame of {video_path.name}")
    
    if results_b:
        print("\nAverage Facial Dimensions:")
        for key in ['face_height_cm', 'face_width_cm', 'eye_width_cm', 'nose_length_cm', 'mouth_width_cm']:
            avg = round(np.mean([r[key] for r in results_b]), 1)
            print(f"  {key.replace('_', ' ').title()}: {avg} cm")
    
    # Save results to folder
    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = results_folder / f"analysis_results_{timestamp}.txt"
    
    with open(result_file, "w") as f:
        f.write("BLINK RATE AND FACIAL DIMENSIONS ANALYSIS\n\n")
        f.write("Task A - Blink Rate:\n")
        for r in results_a:
            f.write(f"\n{r['video']}:\n")
            f.write(f"  Total blinks: {r['total_blinks']}\n")
            f.write(f"  Duration: {r['duration_sec']} seconds\n")
            f.write(f"  Blink rate: {r['blink_rate_per_sec']:.4f} blinks/sec ({r['blink_rate_per_min']:.1f} blinks/min)\n")
        
        if results_a:
            overall = np.mean([r['blink_rate_per_min'] for r in results_a])
            f.write(f"\nOverall average blink rate: {overall:.1f} blinks per minute\n")
        
        f.write("\nTask B - Facial Dimensions (approximate):\n")
        for r in results_b:
            f.write(f"\n{r['video']}:\n")
            f.write(f"  Face height: {r['face_height_cm']} cm\n")
            f.write(f"  Face width:  {r['face_width_cm']} cm\n")
            f.write(f"  Eye width:   {r['eye_width_cm']} cm\n")
            f.write(f"  Nose length: {r['nose_length_cm']} cm\n")
            f.write(f"  Mouth width: {r['mouth_width_cm']} cm\n")
        
        if results_b:
            f.write("\nAverages:\n")
            for key in ['face_height_cm', 'face_width_cm', 'eye_width_cm', 'nose_length_cm', 'mouth_width_cm']:
                avg = round(np.mean([r[key] for r in results_b]), 1)
                f.write(f"  {key.replace('_', ' ').title()}: {avg} cm\n")
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()