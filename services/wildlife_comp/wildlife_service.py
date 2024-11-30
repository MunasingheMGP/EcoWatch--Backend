import cv2
import math
import numpy as np
from ultralytics import YOLO
from services.wildlife_comp.sort import Sort
import pandas as pd
from pathlib import Path
import uuid

BASE_DIR = Path(__file__).resolve().parent
prediction_results = {}

class WildlifeService():
    
    def seconds_to_time_format(self, seconds):
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def process_video(self, video_path, output_video_path, location, date, time):
        task_id = uuid.uuid4()
        cap = cv2.VideoCapture(video_path)
        model = YOLO(f"{BASE_DIR}/weights/best.pt")

        if not cap.isOpened():
            return {"error": "Error opening video file."}

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        classnames = ["Bear", "Deer", "Elephant", "Lion", "Wild boar"]
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        zone = np.array([[20, height - 20], [width - 20, height - 20], 
                        [width - 20, 20], [20, 20]], np.int32)

        tracker = Sort()

        data = {'id': [], 'Frame ID': [], 'Animal Name': [], 'Animal Count': [], 'Frame Time': [], 'Location': [], 'Date': [], 'Time': []}
        df = pd.DataFrame(data)
        id_counter = 1

        # Convert the given time to seconds for processing
        given_time_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(time.split(":"))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            current_detections = np.empty([0, 5])

            animals_inside = []

            for info in results:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    confidence = box.conf[0]

                    if confidence < 0.5:
                        continue

                    class_detect = box.cls[0]
                    class_detect = int(class_detect)
                    class_detect = classnames[class_detect]
                    conf = math.ceil(confidence * 100)
                    cv2.putText(frame, f'{class_detect} {conf}%', (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if class_detect in ['Bear', 'Deer', 'Elephant', 'Lion', 'Wild boar']:
                        detections = np.array([x1, y1, x2, y2, conf])
                        current_detections = np.vstack([current_detections, detections])
                        animals_inside.append(class_detect)

            cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=8)

            track_results = tracker.update(current_detections)

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_time = self.seconds_to_time_format(given_time_seconds + frame_time_seconds)

            for animal in animals_inside:
                df = pd.concat([df, pd.DataFrame({'id': [id_counter], 'Frame ID': [frame_id], 'Animal Name': [animal], 'Animal Count': [len(animals_inside)], 'Frame Time': [frame_time], 'Location': [location], 'Date': [date], 'Time': [time]})], ignore_index=True)
                id_counter += 1

            df.drop_duplicates(subset=['Frame ID', 'Animal Name'], keep='first', inplace=True)

            cv2.putText(frame, f'Total Animals Inside Zone: {len(animals_inside)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(frame)

            # Comment out or remove the lines below to stop displaying the video
            cv2.imshow('Video Processing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_base_name = Path(video_path).stem
        excel_filename = f"{video_base_name}_animal_counts.xlsx"
        df.to_excel(excel_filename, index=False)        

        out.release()
        cap.release()
        cv2.destroyAllWindows()

        # prediction_results[task_id] = {
        #     "video": output_video_path,
        #     "csv": excel_filename
        # }
        prediction_results[task_id] = {
            "video": video_path,
            "output": output_video_path,
            "csv": excel_filename
        }

        return str(task_id)
    
    def get_results(self, task_id:uuid.UUID):
        results = prediction_results.get(task_id, None)
        if results is None:
            return {"error": "Task ID not found"}
        return results