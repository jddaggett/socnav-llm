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

def follow_flow(distance_ahead_m: float) -> Action:
    if distance_ahead_m < 1.0:  return ACTIONS["FORWARD_SLOW"]
    if distance_ahead_m < 1.8:  return ACTIONS["FORWARD_MED"]
    return ACTIONS["FORWARD_FAST"]

def yield_to_human(clearance_m: float) -> Action:
    return ACTIONS["STOP"] if clearance_m < 2.0 else ACTIONS["FORWARD_SLOW"]

def bypass_group(go_left: bool, clearance_ok: bool) -> Action:
    if not clearance_ok: return ACTIONS["STOP"]
    return ACTIONS["STRAFE_LEFT"] if go_left else ACTIONS["STRAFE_RIGHT"]