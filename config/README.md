# Config: Actions, Primitives, and Defaults

This folder defines the **first-order motion primitives**, **second-order social actions**, and **context defaults** used by SocNav-LLM. The **mapping from actions → primitive sequences** is implemented in Python (`second2first_map.py`) for readability, tests, and version control.

## Files

- `first_order_motion_prims.json`  
  Names and parameters of atomic motion commands used in the sim (e.g., `FORWARD_MED`, `TURN_LEFT`). Each primitive specifies `{vx, vy, wz, stop}` in body-frame units.

- `second_order_actions.json`  
  The high-level, semantically meaningful “actions” (tiny policies) that a VLM predicts (e.g., `FOLLOW_FLOW`, `YIELD_TO_HUMAN`, `BYPASS_GROUP_LEFT`). Each action lists required observations and tunable parameters.

- `context_defaults.json`  
  Project-wide constants and thresholds (e.g., `normal_following_distance_m`, `crosswalk_gap_threshold_sec`, `weave_step_duration_s`, speed tiers).

- `second2first_map.py`  
  Code that loads the above JSONs and maps a `(action_name, observations)` pair to a short **sequence** of first-order primitives, each with a duration. 
