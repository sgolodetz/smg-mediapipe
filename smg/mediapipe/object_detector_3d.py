import cv2
import mediapipe as mp
import numpy as np

from scipy.spatial.transform import Rotation as R
from timeit import default_timer as timer
from typing import List, Optional, Tuple

from smg.utility import GeometryUtil


class ObjectDetector3D:
    """A 3D object detector based on MediaPipe Objectron."""

    # NESTED TYPES

    class Object3D:
        """A detected 3D object."""

        # CONSTRUCTOR

        def __init__(self, landmarks_3d: List[np.ndarray]):
            # TODO
            self.__landmarks_3d: List[np.ndarray] = landmarks_3d

        # PROPERTIES

        @property
        def landmarks_3d(self) -> List[np.ndarray]:
            # TODO
            return self.__landmarks_3d

    # CONSTRUCTOR

    def __init__(self, *, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float]):
        # TODO: Comment here.
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        fx, fy, cx, cy = intrinsics

        # TODO: Comment here.
        self.__objectron: mp.solutions.objectron.Objectron = mp.solutions.objectron.Objectron(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_name="Chair",
            focal_length=(fx, fy),
            principal_point=(cx, cy),
            image_size=image_size
        )

    # PUBLIC METHODS

    def detect_objects(self, image: np.ndarray, world_from_camera: np.ndarray) -> List[Object3D]:
        # TODO
        objects: List[ObjectDetector3D.Object3D] = []

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

                camera_landmarks_3d: List[np.ndarray] = []
                landmarks_3d: List[np.ndarray] = []

                m: np.ndarray = np.eye(4)
                m[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * np.pi).as_matrix()
                world_from_camera = world_from_camera @ m

                for landmark_3d in detected_object.landmarks_3d.landmark:
                    camera_landmark_3d: np.ndarray = np.array([landmark_3d.x, landmark_3d.y, landmark_3d.z])
                    camera_landmarks_3d.append(camera_landmark_3d)

                scale: float = ObjectDetector3D.__calculate_scale(camera_landmarks_3d[1], world_from_camera)
                for camera_landmark_3d in camera_landmarks_3d:
                    landmarks_3d.append(
                        GeometryUtil.apply_rigid_transform(world_from_camera, scale * camera_landmark_3d)
                    )

                objects.append(ObjectDetector3D.Object3D(landmarks_3d))

        cv2.imshow("Annotated Image", cv2.resize(annotated_image, (640, 480)))
        cv2.waitKey(1)

        # TODO
        return objects

    # PRIVATE STATIC METHODS

    @staticmethod
    def __calculate_scale(ground_landmark: np.ndarray, world_from_camera: np.ndarray) -> float:
        start = timer()

        alpha: float = 1.5
        scale: float = 1.0
        transformed_ground_landmark: np.ndarray = GeometryUtil.apply_rigid_transform(world_from_camera, ground_landmark)
        while np.fabs(transformed_ground_landmark[1]) > 0.01:
            err: float = transformed_ground_landmark[1]
            print(scale, err)
            scale -= alpha * err
            transformed_ground_landmark = GeometryUtil.apply_rigid_transform(world_from_camera, scale * ground_landmark)

        end = timer()
        print(f"{end - start}s")

        return scale
