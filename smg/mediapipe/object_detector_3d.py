import cv2
import mediapipe as mp
import numpy as np

# noinspection PyPep8Naming
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
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

    def __init__(self, *, debug: bool = True, image_size: Tuple[int, int],
                 intrinsics: Tuple[float, float, float, float]):
        """
        TODO

        :param debug:       Whether or not to enable debugging.
        :param image_size:  TODO
        :param intrinsics:  TODO
        """
        # TODO: Comment here.
        self.__debug: bool = debug
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics

        # TODO: Comment here.
        fx, fy, cx, cy = intrinsics
        self.__objectron: mp.solutions.objectron.Objectron = mp.solutions.objectron.Objectron(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_name="Chair",
            focal_length=(fx, fy),
            principal_point=(cx, cy),
            image_size=image_size
        )

    # PUBLIC METHODS

    def detect_objects(self, image: np.ndarray, world_from_camera: np.ndarray, *, angle_threshold: float = 10.0,
                       ground_plane: Tuple[float, float, float, float] = (0.0, -1.0, 0.0, 0.0)) -> List[Object3D]:
        # TODO
        objects: List[ObjectDetector3D.Object3D] = []

        # TODO: Comment here.
        results = self.__objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()
        # noinspection PyUnresolvedReferences
        if results.detected_objects is not None:
            # noinspection PyUnresolvedReferences
            for detected_object in results.detected_objects:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image, detected_object.landmarks_2d, mp.solutions.objectron.BOX_CONNECTIONS
                )
                mp.solutions.drawing_utils.draw_axis(
                    annotated_image, detected_object.rotation, detected_object.translation
                )

                camera_landmarks_3d: List[np.ndarray] = []
                landmarks_3d: List[np.ndarray] = []

                for landmark_3d in detected_object.landmarks_3d.landmark:
                    camera_landmark_3d: np.ndarray = np.array([landmark_3d.x, -landmark_3d.y, -landmark_3d.z])
                    camera_landmarks_3d.append(camera_landmark_3d)
                    landmarks_3d.append(
                        GeometryUtil.apply_rigid_transform(world_from_camera, camera_landmark_3d)
                    )

                cam: SimpleCamera = CameraPoseConverter.pose_to_camera(np.linalg.inv(world_from_camera))
                for i in [1, 2, 5, 6]:
                    landmarks_3d[i] = GeometryUtil.find_plane_intersection(
                        cam.p(), landmarks_3d[i] - cam.p(), ground_plane
                    )
                    landmarks_3d[i + 2] = landmarks_3d[i] + np.array([0, -1, 0])

                landmarks_3d[0] = np.mean([landmarks_3d[i] for i in [1, 2, 5, 6]], axis=0) + np.array([0, -0.5, 0])

                angle1: Optional[float] = (180 / np.pi) * GeometryUtil.angle_between(
                    landmarks_3d[2] - landmarks_3d[1], landmarks_3d[5] - landmarks_3d[1]
                )
                angle2: Optional[float] = (180 / np.pi) * GeometryUtil.angle_between(
                    landmarks_3d[2] - landmarks_3d[6], landmarks_3d[5] - landmarks_3d[6]
                )
                if self.__debug:
                    print(angle1, angle2)
                if angle1 is not None and angle2 is not None \
                        and np.fabs(angle1 - 90) <= angle_threshold and np.fabs(angle2 - 90) <= angle_threshold:
                    objects.append(ObjectDetector3D.Object3D(landmarks_3d))

        if self.__debug:
            cv2.imshow("Annotated Image", cv2.resize(annotated_image, (640, 480)))
            cv2.waitKey(1)

        # TODO
        return objects
