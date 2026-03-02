from typing import Dict, List

SMPLH_PRIMARY_JOINTS: Dict[str, int] = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


FRAME_FALLBACKS: Dict[str, List[str]] = {
    "pelvis": ["pelvis", "base_link", "base", "Waist", "waist_link", "trunk", "Trunk"],
    "torso_link": ["torso_link", "torso", "spine", "waist_pitch_link", "waist_roll_link"],
    "left_hip_roll_link": ["left_hip_roll_link", "left_hip_yaw_link", "left_hip_pitch_link", "left_hip_link"],
    "left_knee_link": ["left_knee_link"],
    "left_toe_link": ["left_toe_link", "left_foot_link", "left_ankle_roll_link", "left_ankle_pitch_link"],
    "right_hip_roll_link": ["right_hip_roll_link", "right_hip_yaw_link", "right_hip_pitch_link", "right_hip_link"],
    "right_knee_link": ["right_knee_link"],
    "right_toe_link": ["right_toe_link", "right_foot_link", "right_ankle_roll_link", "right_ankle_pitch_link"],
    "left_shoulder_yaw_link": ["left_shoulder_yaw_link", "left_shoulder_roll_link", "left_shoulder_pitch_link"],
    "left_elbow_link": ["left_elbow_link"],
    "left_wrist_yaw_link": ["left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"],
    "right_shoulder_yaw_link": ["right_shoulder_yaw_link", "right_shoulder_roll_link", "right_shoulder_pitch_link"],
    "right_elbow_link": ["right_elbow_link"],
    "right_wrist_yaw_link": ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"],
}

BASE_BODY_CANDIDATES: List[str] = [
    "pelvis",
    "base_link",
    "base",
    "Waist",
    "waist_link",
    "trunk",
    "Trunk",
    "torso",
]
