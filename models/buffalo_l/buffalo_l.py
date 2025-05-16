import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

def detect_faces_with_buffalo_l(image_path):
    """
    Detect faces in an image using InsightFace's Buffalo_L model
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Original image with face detection bounding boxes and landmarks
    """
    # Initialize the FaceAnalysis application
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    
    # Set the buffalo_l model as the face detection model
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = app.get(img)
    
    # Draw results on the image
    for face in faces:
        # Get the bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw facial landmarks
        if hasattr(face, 'landmark_2d_106'):
            landmark = face.landmark_2d_106.astype(np.int32)
            for pt in landmark:
                cv2.circle(img, (pt[0], pt[1]), 1, (0, 0, 255), -1)
        
        # Add confidence score text
        confidence = face.det_score
        text = f"Conf: {confidence:.2f}"
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img, faces

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "./image1.png"
    
    # Detect faces
    result_img, detected_faces = detect_faces_with_buffalo_l(image_path)
    
    # Print number of faces detected
    print(f"Number of faces detected: {len(detected_faces)}")
    
    #Display the result
    # plt.figure(figsize=(12, 8))
    # plt.imshow(result_img)
    # plt.axis('off')
    # plt.title('Face Detection with InsightFace Buffalo_L')
    # plt.show()
    #保存图像
    cv2.imwrite("detected_faces.png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    # Print additional face information
    for i, face in enumerate(detected_faces):
        print(f"Face {i+1}:")
        print(f"  Confidence: {face.det_score:.4f}")
        print(f"  Bounding Box: {face.bbox.astype(int)}")
        print(f"  Age: {face.age if hasattr(face, 'age') else 'N/A'}")
        print(f"  Gender: {face.gender if hasattr(face, 'gender') else 'N/A'}")
        print()