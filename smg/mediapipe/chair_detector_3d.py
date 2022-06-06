import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d

from typing import List, Optional, Tuple

from smg.open3d import VisualisationUtil
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import GeometryUtil


class ChairDetector3D:
    """A 3D chair detector based on MediaPipe Objectron for chairs sitting on the ground."""

    # NESTED TYPES

    class Chair:
        """A detected chair."""

        # CONSTRUCTOR

        def __init__(self, landmarks_3d: List[np.ndarray]):
            """
            Construct a detected chair based on the specified 3D landmarks.

            .. note::
                The first 3D landmark denotes the centre of a bounding box for the chair, and the
                remaining 3D landmarks denote the corners of that box.

            :param landmarks_3d:    The 3D landmarks specifying the detected chair.
            """
            self.__landmarks_3d: List[np.ndarray] = landmarks_3d

        # PROPERTIES

        @property
        def landmarks_3d(self) -> List[np.ndarray]:
            """
            Get the 3D landmarks associated with the detected chair.

            :return:    The 3D landmarks associated with the detected chair.
            """
            return self.__landmarks_3d

        # PUBLIC METHODS

        def make_o3d_geometries(self, *, corner_colours: List[Tuple[float, float, float]],
                                edge_colour: Tuple[float, float, float]) -> List[o3d.geometry.Geometry]:
            """
            Make the Open3D geometries needed to visualise the detected chair's bounding box.

            :param corner_colours:  The colours to assign to the corners of the bounding box.
            :param edge_colour:     The colour to assign to the edges of the bounding box.
            :return:                The Open3D geometries needed to visualise the bounding box.
            """
            geometries: List[o3d.geometry.Geometry] = []

            # Add a sphere for each corner of the bounding box.
            for i, landmark_3d in enumerate(self.__landmarks_3d):
                geometries.append(VisualisationUtil.make_sphere(landmark_3d, 0.01, colour=corner_colours[i]))

            # Add the edges of the bounding box.
            edge_indices: np.ndarray = np.array([
                [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4], [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8]
            ])
            edges: o3d.geometry.LineSet = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(self.__landmarks_3d),
                lines=o3d.utility.Vector2iVector(edge_indices),
            )
            edges.colors = o3d.utility.Vector3dVector([edge_colour for _ in range(len(edge_indices))])
            geometries.append(edges)

            return geometries

    # CONSTRUCTOR

    def __init__(
        self, *, debug: bool = False, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float],
        min_detection_confidence: float = 0.5, static_image_mode: bool = True
    ):
        """
        Construct a 3D chair detector based on MediaPipe Objectron for chairs sitting on the ground.

        :param debug:                       Whether or not to enable debugging.
        :param image_size:                  The image size, as a (width, height) tuple.
        :param intrinsics:                  The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :param min_detection_confidence:    The minimum confidence value (in [0,1]) for a detection to be retained.
        :param static_image_mode:           Whether to treat the input images as individual images or a video stream.
        """
        self.__debug: bool = debug
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics

        # Construct the Objectron model.
        fx, fy, cx, cy = intrinsics
        self.__objectron: mp.solutions.objectron.Objectron = mp.solutions.objectron.Objectron(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            model_name="Chair",
            focal_length=(fx, fy),
            principal_point=(cx, cy),
            image_size=image_size
        )

    # PUBLIC METHODS

    def detect_chairs(self, image: np.ndarray, world_from_camera: np.ndarray, *, angle_threshold: float = 10.0,
                      ground_plane: Tuple[float, float, float, float] = (0.0, -1.0, 0.0, 0.0)) -> List[Chair]:
        """
        Try to detect any 3D chairs in the specified image.

        :param image:               The specified image (assumed to be in BGR format, as per OpenCV).
        :param world_from_camera:   The camera pose.
        :param angle_threshold:     The maximum angle (in degrees) by which the corner angles at the base of the
                                    bounding box for a detected chair are allowed to deviate from 90 degrees. We
                                    use this to suppress Objectron detections that are too far from being cuboids.
        :param ground_plane:        The ground plane, as an (a,b,c,d) tuple denoting the plane ax + by + cz - d = 0.
                                    We project the base landmarks of the bounding boxes produced by Objectron into
                                    this plane, as Objectron by default produces boxes that are floating in mid-air.
        :return:                    A list containing any chairs detected in the image.
        """
        objects: List[ChairDetector3D.Chair] = []

        # Run the Objectron model on the image, after first converting it to RGB format.
        results = self.__objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # If we're debugging, make a copy of the image onto which to draw helpful annotations.
        annotated_image: Optional[np.ndarray] = image.copy() if self.__debug else None

        # If any chairs were detected in the image:
        # noinspection PyUnresolvedReferences
        if results.detected_objects is not None:
            # For each detected chair:
            # noinspection PyUnresolvedReferences
            for detected_chair in results.detected_objects:
                # If we're debugging:
                if self.__debug:
                    # Draw the 2D landmarks for the chair onto the annotated image.
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image, detected_chair.landmarks_2d, mp.solutions.objectron.BOX_CONNECTIONS
                    )

                    # Draw the coordinate system for the chair onto the annotated image.
                    mp.solutions.drawing_utils.draw_axis(
                        annotated_image, detected_chair.rotation, detected_chair.translation
                    )

                # Transform the 3D landmarks of the detected chair into world space.
                landmarks_3d: List[np.ndarray] = []

                # For each landmark:
                for landmark_3d in detected_chair.landmarks_3d.landmark:
                    # Determine the landmark's position in camera space. The Objectron coordinate system uses
                    # x=right, y=up, z=back, as opposed to our convention of x=right, y=down, z=forward, so we
                    # negate the y and z coordinates of the landmark to account for this.
                    camera_landmark_3d: np.ndarray = np.array([landmark_3d.x, -landmark_3d.y, -landmark_3d.z])

                    # Transform the landmark's position into world space and add it to the list.
                    landmarks_3d.append(
                        GeometryUtil.apply_rigid_transform(world_from_camera, camera_landmark_3d)
                    )

                # Project the base landmarks of the bounding box into the ground plane, and update the other
                # landmarks accordingly.
                cam: SimpleCamera = CameraPoseConverter.pose_to_camera(np.linalg.inv(world_from_camera))
                base_landmark_indices: List[int] = [1, 2, 5, 6]
                for i in base_landmark_indices:
                    landmarks_3d[i] = GeometryUtil.find_plane_intersection(
                        cam.p(), landmarks_3d[i] - cam.p(), ground_plane
                    )
                    landmarks_3d[i + 2] = landmarks_3d[i] + np.array([0, -1, 0])

                landmarks_3d[0] = np.array([0, -0.5, 0]) + np.mean(
                    [landmarks_3d[i] for i in base_landmark_indices], axis=0
                )

                # Calculate the corner angles at the base of the bounding box (in degrees).
                angle1: Optional[float] = (180 / np.pi) * GeometryUtil.angle_between(
                    landmarks_3d[2] - landmarks_3d[1], landmarks_3d[5] - landmarks_3d[1]
                )
                angle2: Optional[float] = (180 / np.pi) * GeometryUtil.angle_between(
                    landmarks_3d[2] - landmarks_3d[6], landmarks_3d[5] - landmarks_3d[6]
                )

                # If we're debugging, print the angles.
                if self.__debug:
                    print(f"- Angle #1: {angle1}; Angle #2: {angle2}")

                # If both of the corner angles are no greater than the threshold:
                if angle1 is not None and angle2 is not None \
                        and np.fabs(angle1 - 90) <= angle_threshold and np.fabs(angle2 - 90) <= angle_threshold:
                    # Add the detected chair to the list.
                    objects.append(ChairDetector3D.Chair(landmarks_3d))

        # If we're debugging, show the annotated image, after resizing it to fit on the screen.
        if self.__debug:
            cv2.imshow("Annotated Image", cv2.resize(annotated_image, (640, 480)))
            cv2.waitKey(1)

        return objects
