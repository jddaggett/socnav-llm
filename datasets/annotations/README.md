# Manual Annotations

This folder contains manually created fine-grained annotations for selected **SCAND** sped-up trajectory videos (.mp4 or .avi) used in this project.

Each annotation describes a short segment (typically 3–10 s) of a video where the robot exhibits a distinct **socially compliant navigation behavior**.

---

## `scand_manual_annotations.csv` Format

| Column | Description |
|---------|-------------|
| `video_file` | Name of the annotated SCAND video file (from `../videos/*`) |
| `start_time_s` | Start time of the annotated segment in seconds |
| `end_time_s` | End time of the annotated segment in seconds |
| `robot_type` | Robot platform used during data collection (`Spot` or `Jackal`) |
| `environment_type` | General environment category (`indoor`, `outdoor`, `semi-outdoor`) |
| `crowd_density` | Approximate pedestrian density during the segment (`low`, `medium`, `high`) |
| `observed_behavior` | Short free-text description of what the robot did |
| `social_label` | One or more standardized SCAND social navigation labels (see below) |
| `notes` | Optional commentary, e.g., scene context or reasoning |

---

## Social Labels
*(Adapted from the official SCAND metadata: `../metadata/AB_Readme_SCANDv2.pdf`)*

| Label | Description |
|--------|-------------|
| **Against Traffic** | Navigating against oncoming pedestrian flow |
| **With Traffic** | Moving with surrounding pedestrian flow |
| **Street Crossing** | Crossing a street or intersection |
| **Overtaking** | Passing a person or group moving in same direction |
| **Sidewalk** | Navigating along a sidewalk |
| **Passing Conversational Groups** | Passing nearby stationary groups talking among themselves |
| **Blind Corner** | Turning or navigating past a corner with low visibility |
| **Narrow Doorway** | Passing through a narrow doorway, often waiting for a human to open it |
| **Crossing Stationary Queue** | Moving across a line of waiting people |
| **Stairs** | Ascending or descending stairs |
| **Vehicle Interaction** | Navigating near or around a parked or moving vehicle |
| **Navigating Through Large Crowds** | Moving through dense, unstructured pedestrian crowds |

---



