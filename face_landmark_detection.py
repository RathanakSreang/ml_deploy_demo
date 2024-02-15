import cv2
import face_recognition
import numpy as np
import time


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(
        face_distance(known_face_encodings, face_encoding_to_check) <= tolerance
    )


def cosine_similarity(X, Y=None, dense_output=True):
    X_normalized = X / np.linalg.norm(X, ord=2)
    Y_normalized = Y / np.linalg.norm(Y, ord=2, keepdims=True)
    return X_normalized.dot(Y_normalized.T)


def get_face_embbed(rgb_small_frame):
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    return face_encodings


my_face = cv2.imread("0001.jpg")
my_face_encodings = get_face_embbed(my_face)[0]
print(cosine_similarity(my_face_encodings, my_face_encodings))
print(face_distance([my_face_encodings], my_face_encodings))
print(compare_faces([my_face_encodings], my_face_encodings))

the_rock = cv2.imread("0002.jpg")
the_rock_encodings = get_face_embbed(the_rock)[0]
print(cosine_similarity(my_face_encodings, the_rock_encodings))
print(face_distance([my_face_encodings], the_rock_encodings))
print(compare_faces([my_face_encodings], the_rock_encodings))

