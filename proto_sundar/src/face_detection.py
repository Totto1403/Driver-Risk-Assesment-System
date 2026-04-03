from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(frame):
    results = detector.detect_faces(frame)
    
    faces = []
    for result in results:
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)
        faces.append((x, y, w, h))
    
    return faces