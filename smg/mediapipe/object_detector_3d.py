import cv2
import mediapipe as mp
import numpy as np

from typing import List, Tuple


class ObjectDetector3D:
    """A 3D object detector based on MediaPipe Objectron."""

    # NESTED TYPES

    class Object3D:
        """A detected 3D object."""
        # TODO
        pass

    # CONSTRUCTOR

    def __init__(self, *, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float]):
        # TODO: Comment here.
        fx, fy, cx, cy = intrinsics

        # TODO: Comment here.
        self.__objectron: mp.solutions.objectron.Objectron = mp.solutions.objectron.Objectron(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_name="Chair",
            focal_length=(fx, fy),
            principal_point=(cx, cy),
            image_size=image_size
        )

    # PUBLIC METHODS

    def detect_objects(self, image: np.ndarray) -> List[Object3D]:
        # TODO: Comment here.
        results = self.__objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # TODO: Comment here.
        print(results.detected_objects)

        annotated_image = image.copy()
        if results.detected_objects is not None:
            for detected_object in results.detected_objects:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image, detected_object.landmarks_2d, mp.solutions.objectron.BOX_CONNECTIONS
                )
                mp.solutions.drawing_utils.draw_axis(
                    annotated_image, detected_object.rotation, detected_object.translation
                )

        cv2.imshow("Annotated Image", cv2.resize(annotated_image, (640, 480)))
        cv2.waitKey()

        # TODO
        return []
