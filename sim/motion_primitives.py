from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    vx: float   # forward m/s (+ forward)
    vy: float   # lateral m/s (+ left)
    wz: float   # yaw rad/s (+ CCW)
    stop: bool = False

ACTIONS = {
    "STOP":          Action(0.0,  0.0,  0.0, stop=True),
    "FORWARD_SLOW":  Action(0.30, 0.0,  0.0),
    "FORWARD_MED":   Action(0.60, 0.0,  0.0),
    "FORWARD_FAST":  Action(1.00, 0.0,  0.0),
    "STRAFE_LEFT":   Action(0.20, 0.30, 0.0),
    "STRAFE_RIGHT":  Action(0.20,-0.30, 0.0),
    "TURN_LEFT":     Action(0.0,  0.0,  0.8),
    "TURN_RIGHT":    Action(0.0,  0.0, -0.8),
    "BACK_OFF":      Action(-0.25,0.0,  0.0),
}