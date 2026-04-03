import cv2
import time
import proj.utils as utils
from proj.analyzers import DrowsinessDetector, FerAnalyzer, ObjectDetector, PoseAnalyzer
from proj.logic import DriverRiskLogic

FER_MODEL_PATH = "models/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt" 
WEBCAM_ID = 0

def main():
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {WEBCAM_ID}")
        return

    # Lowered MAR threshold for easier yawning demo
    drowsiness_detector = DrowsinessDetector(mar_threshold=0.30)
    object_detector = ObjectDetector()  
    pose_analyzer = PoseAnalyzer()
    
    try:
        fer_analyzer = FerAnalyzer(model_path=FER_MODEL_PATH)
    except Exception as e:
        print(e)
        return

    driver_logic = DriverRiskLogic()

    alert_message = ""
    alert_reason = ""
    alert_end_time = 0
    ALERT_DURATION_SEC = 3 

    frame_count = 0
    total_fps = 0
    total_fer_latency = 0 
    total_mp_latency = 0  
    avg_fps = 0
    avg_fer_latency = 0   
    avg_mp_latency = 0    
 
    print("Starting Driver Risk Assessment System... Press 'q' to quit.")

    while cap.isOpened():
        loop_start_time = time.perf_counter()
        
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)

        mp_start_time = time.perf_counter()
        
        drowsiness_result = drowsiness_detector.process(frame)
        
        pose_results = pose_analyzer.process_frame(frame)
        pose_analyzer.draw_landmarks(frame, pose_results)
        pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
        
        mp_latency_ms = (time.perf_counter() - mp_start_time) * 1000
        total_mp_latency += mp_latency_ms
        
        phone_detected, phone_bboxes = object_detector.process(frame)
        
        is_alert_active = time.time() < alert_end_time
        results = None
        emotion_label = "Neutral"

        if drowsiness_result["face_detected"]:
            landmarks = drowsiness_result["landmarks"]
            
            face_crop = utils.get_face_crop(frame, landmarks)
            if face_crop is not None and face_crop.size > 0:
                fer_latency_start = time.perf_counter()
                face_tensor = utils.preprocess_fer(face_crop) 
                emotion_label = fer_analyzer.predict(face_tensor)
                total_fer_latency += (time.perf_counter() - fer_latency_start) * 1000

            results = driver_logic.update(
                drowsiness_result, 
                pose_landmarks, 
                frame.shape, 
                emotion_label, 
                phone_detected
            )

            if results["play_alert"]:
                utils.play_alert() 
                alert_message = results["status"]
                alert_reason = results["reason"]
                alert_end_time = time.time() + ALERT_DURATION_SEC
                is_alert_active = True

            draw_overlay(frame, results, is_alert_active, alert_message, alert_reason, avg_fps, avg_fer_latency, avg_mp_latency)
            
            for (x1, y1, x2, y2) in phone_bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        else:
            cv2.putText(frame, "No driver detected!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_count += 1
        total_fps += 1.0 / (time.perf_counter() - loop_start_time)

        if frame_count % 10 == 0:
            avg_fps = total_fps / 10
            avg_fer_latency = total_fer_latency / 10 
            avg_mp_latency = total_mp_latency / 10   
            total_fps, total_fer_latency, total_mp_latency = 0, 0, 0
        
        cv2.imshow('Driver Risk Assessment System', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


def draw_overlay(frame, results, is_alert_active, alert_message, alert_reason, fps, fer_latency, mp_latency):
    if results:
        score = results['risk_score']
        status = results['status']
        reason_text = results.get('reason', '')
        
        if score < 25: color = (0, 255, 0)       
        elif score < 50: color = (0, 255, 255)   
        else: color = (0, 0, 255)                

        cv2.rectangle(frame, (10, 10), (210, 40), (0,0,0), -1)
        cv2.rectangle(frame, (10, 10), (10 + min(score * 2, 200), 40), color, -1)
        cv2.putText(frame, f"RISK: {score}%", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"STATUS: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw Reason beneath status if it exists
        if reason_text:
            cv2.putText(frame, f"REASON: {reason_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Shifted these down so they don't overlap the new Reason text
        emotion = results['emotion']
        emotion_color = (0, 0, 255) if emotion in ["Angry", "Fear"] else (255, 255, 0)
        cv2.putText(frame, f"EMOTION: {emotion}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        cv2.putText(frame, f"YAWNS (60s): {results['yawns']}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255) if results['yawns'] > 0 else (255, 255, 255), 2)
                    
        posture_col = (0, 0, 255) if results['posture'] in ["Slouching", "Leaning"] else (0, 255, 0)
        cv2.putText(frame, f"BODY: {results['posture']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_col, 2)
        
        cv2.putText(frame, f"Yaw: {results['yaw']:.1f}", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Pitch: {results['pitch']:.1f}", (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Roll: {results['roll']:.1f}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FER Latency: {fer_latency:.0f} ms", (frame.shape[1] - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MP Latency: {mp_latency:.0f} ms", (frame.shape[1] - 220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if is_alert_active:
            if int(time.time() * 2) % 2 == 0:
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                cv2.rectangle(frame, (center_x - 300, center_y - 60), (center_x + 300, center_y + 60), (0, 0, 255), -1)
                cv2.rectangle(frame, (center_x - 300, center_y - 60), (center_x + 300, center_y + 60), (0, 0, 0), 2)
                
                # Draw Alert and the Reason inside the flashing box
                cv2.putText(frame, alert_message, (center_x - 280, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"[{alert_reason}]", (center_x - 280, center_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

if __name__ == "__main__":
    main()