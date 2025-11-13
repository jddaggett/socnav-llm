## behavior_map.py
## This is a module that defines a mapping of behavior to motion primitives for a robot.

BEHAVIORS = {
    "FOLLOW_PERSON": [
        {"action": "FORWARD_SLOW", "duration": 1.0}
    ],

    "PASS_LEFT": [
        {"action": "STRAFE_LEFT",  "duration": 1.2},
        {"action": "FORWARD_MED",  "duration": 1.8}
    ],

    "PASS_RIGHT": [
        {"action": "STRAFE_RIGHT", "duration": 1.2},
        {"action": "FORWARD_MED",  "duration": 1.8}
    ],

    "YIELD": [
        {"action": "STOP",         "duration": 1.5}
    ],

    "APPROACH_INTERACTION": [
        {"action": "FORWARD_SLOW", "duration": 0.8},
        {"action": "STOP",         "duration": 0.5}
    ],

    "BACK_OFF_SAFETY": [
        {"action": "BACK_OFF",     "duration": 1.5}
    ]
}