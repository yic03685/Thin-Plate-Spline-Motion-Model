import cv2
import dlib
import sys

def detect_and_track_face(input_video, output_video):
    # Initialize face detector and shape predictor
    face_detector = dlib.get_frontal_face_detector()

    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # Exponential moving average (EMA) for smoother bounding box
    alpha_location = 0.1
    alpha_dimensions = 0.01
    ema_location = None
    ema_dimensions = None
    output_size = None
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)


        # Update face bounding box if a face is detected
        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            center_x, center_y = x + w // 2, y + h // 2

            # Update EMA for bounding box
            # Update EMA for bounding box location (center) and dimensions
            if ema_location is None:
                ema_location = (center_x, center_y)
                ema_dimensions = (w, h)
            else:
                ema_location = (
                    alpha_location * center_x + (1 - alpha_location) * ema_location[0],
                    alpha_location * center_y + (1 - alpha_location) * ema_location[1]
                )
                ema_dimensions = (
                    alpha_dimensions * w + (1 - alpha_dimensions) * ema_dimensions[0],
                    alpha_dimensions * h + (1 - alpha_dimensions) * ema_dimensions[1]
                )

        # Use the last detected face bounding box if no face is detected
        if ema_location is not None and ema_dimensions is not None:
            # Calculate the square bounding box with 10% margin
            max_dim = int(max(ema_dimensions[0], ema_dimensions[1]) * 2)

            if output_size is None:
                output_size = (max_dim, max_dim)

            x1 = max(0, int(ema_location[0] - max_dim // 2))
            y1 = max(0, int(ema_location[1] - max_dim // 2))
            x2 = min(x1 + max_dim, frame.shape[1])
            y2 = min(y1 + max_dim, frame.shape[0])

            # Crop the face and pad to maintain a consistent size
            cropped_face = frame[y1:y2, x1:x2]

            if output_size is None:
                pass
            else:
                cropped_face = cv2.resize(cropped_face, output_size)

            print(cropped_face.shape)

            # Set the output video size based on the first cropped face
            if out is None:
                out = cv2.VideoWriter(output_video, fourcc, fps, output_size)

            # Write the cropped face to the output video
            out.write(cropped_face)
            cv2.imshow("Cropped Face", cropped_face)

    # Release the video capture, video writer, and close all windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    input_video = "lawrence_original.mp4"
    output_video = "tracked_face_6.mp4"

    detect_and_track_face(input_video, output_video)