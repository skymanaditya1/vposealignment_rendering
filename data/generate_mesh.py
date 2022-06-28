# import mediapipe as mp
import cv2
import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# draw the landmarks on top of the image 
def drawPolyline(image, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks[i][0], landmarks[i][1]]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)

# Draw lines around landmarks corresponding to different facial regions
def drawPolylines(image, landmarks):
    drawPolyline(image, landmarks, 0, 16)           # Jaw line
    drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(image, landmarks, 27, 30)          # Nose bridge
    drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(image, landmarks, 36, 41, True)    # Left eye
    drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

# generate the 3d landmarks directly from the image
def landmark_from_image(frame, fa, require_image=False):
    landmarks = fa.get_landmarks_from_image(frame) # check if this is using the GPU

    # print(f'Device used inside the landmark generation code : {fa.device}')

    # generate a white image with landmarks drawn if required 
    if require_image:
        white_image = np.ones(frame.shape, dtype=np.uint8) * 255
        drawPolylines(white_image, landmarks[0])

        return white_image

    return landmarks

def landmark_from_batch(batch_tensor, fa):
    landmarks = fa.get_landmarks_from_batch(batch_tensor)
    num_landmarks = 68

    mask = [True if len(landmarks[i]) == 0 else False for i in range(len(landmarks))]

    # custom code to handle cases where not all landmarks are detected
    # landmarks_np = np.empty((batch_tensor.size(0), num_landmarks, 3))
    # print(f'batch size : {batch_tensor.shape[0]}')
    landmarks_np = np.empty((batch_tensor.shape[0], num_landmarks, 3))

    # print(f'Np landmarks inside generate mesh : {landmarks_np.shape}')

    for i, landmark in enumerate(landmarks):
        if len(landmark) != 0:
            landmarks_np[i] = landmark[:num_landmarks]

    # convert landmarks to torch tensor and also generate a mask

    return landmarks_np, mask

'''
# generate the mesh directly from the image
def mesh_from_image(frame, color_transform=False):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # Convert the BGR image to RGB before processing.
        if color_transform:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            results = face_mesh.process(frame)

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return None

        # annotated_image = image.copy()
        annotated_image = np.ones(frame.shape) * 255
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
    
    return annotated_image

# generate the mesh from the filepath of the image
def mesh_from_filepath(image_path):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        image = cv2.imread(image_path)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return None

        # annotated_image = image.copy()
        annotated_image = np.ones(image.shape) * 255
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
    
    return annotated_image
'''