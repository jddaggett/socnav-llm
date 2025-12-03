"""
Plot proxemics comparison between VLM and RRT motion planners.

This script visualizes the distances maintained between the robot and human agents
for both VLM-based and RRT-based trajectory planning approaches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION 
# ============================================================================

# File paths
RRT_TRAJECTORY_FILE = "rrt_sidewalk_navigation_traj.csv"
VLM_TRAJECTORY_FILE = "vlm_sidewalk_navigation_traj.csv"

# Time scaling
# Speed up RRT trajectory by this factor for visualization
# Set to 1.0 for no scaling, >1.0 to speed up RRT timeline
RRT_TIME_SPEEDUP_FACTOR = 5.0

# Plot titles and labels
PLOT_TITLE = "Proxemics Comparison: VLM vs RRT"
X_AXIS_LABEL = "Time (seconds)"
Y_AXIS_LABEL = "Euclidean Distance to Person (meters)"
LEGEND_LOCATION = "upper right"

# Line colors (matplotlib color names or hex codes)
VLM_PERSON1_COLOR = "#1f77b4"  # Blue
VLM_PERSON2_COLOR = "#ff7f0e"  # Orange
RRT_PERSON1_COLOR = "#2ca02c"  # Green
RRT_PERSON2_COLOR = "#d62728"  # Red

# Line styles
VLM_LINE_STYLE = "-"   # Solid line
RRT_LINE_STYLE = "--"  # Dashed line

# Line widths
VLM_LINE_WIDTH = 2.0
RRT_LINE_WIDTH = 2.0

# Line transparency (0.0 = transparent, 1.0 = opaque)
VLM_ALPHA = 0.8
RRT_ALPHA = 0.8

# Labels for legend
VLM_PERSON1_LABEL = "VLM - Person 1"
VLM_PERSON2_LABEL = "VLM - Person 2"
RRT_PERSON1_LABEL = "RRT (sped up) - Person 1"
RRT_PERSON2_LABEL = "RRT (sped up) - Person 2"

# Figure size (width, height in inches)
FIGURE_SIZE = (12, 7)

# Grid settings
SHOW_GRID = True
GRID_ALPHA = 0.3
GRID_LINE_STYLE = ':'

# Font sizes
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11

# Output settings
SAVE_PLOT = True
OUTPUT_FILE = "proxemics_comparison.png"
OUTPUT_DPI = 300

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def calculate_euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def load_and_process_trajectory(filepath, planner_name):
    """Load trajectory CSV and calculate distances to each person."""
    df = pd.read_csv(filepath)
    
    # Calculate distance to person 1
    df[f'{planner_name}_dist_person1'] = calculate_euclidean_distance(
        df['robot_x'], df['robot_y'],
        df['person1_slide_x'], df['person1_slide_y']
    )
    
    # Calculate distance to person 2
    df[f'{planner_name}_dist_person2'] = calculate_euclidean_distance(
        df['robot_x'], df['robot_y'],
        df['person2_slide_x'], df['person2_slide_y']
    )
    
    return df


def main():
    """Main function to create the proxemics comparison plot."""
    
    # Load data
    print(f"Loading {RRT_TRAJECTORY_FILE}...")
    rrt_df = load_and_process_trajectory(RRT_TRAJECTORY_FILE, 'rrt')
    print(f"  RRT trajectory: {len(rrt_df)} steps, {rrt_df['time'].max():.2f}s duration")
    
    print(f"Loading {VLM_TRAJECTORY_FILE}...")
    vlm_df = load_and_process_trajectory(VLM_TRAJECTORY_FILE, 'vlm')
    print(f"  VLM trajectory: {len(vlm_df)} steps, {vlm_df['time'].max():.2f}s duration")
    
    # Apply time speedup to RRT trajectory
    if RRT_TIME_SPEEDUP_FACTOR != 1.0:
        rrt_df['time'] = rrt_df['time'] / RRT_TIME_SPEEDUP_FACTOR
        print(f"\nApplying {RRT_TIME_SPEEDUP_FACTOR}x speedup to RRT trajectory")
        print(f"  RRT adjusted duration: {rrt_df['time'].max():.2f}s")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot VLM trajectories
    ax.plot(vlm_df['time'], vlm_df['vlm_dist_person1'],
            color=VLM_PERSON1_COLOR,
            linestyle=VLM_LINE_STYLE,
            linewidth=VLM_LINE_WIDTH,
            alpha=VLM_ALPHA,
            label=VLM_PERSON1_LABEL)
    
    ax.plot(vlm_df['time'], vlm_df['vlm_dist_person2'],
            color=VLM_PERSON2_COLOR,
            linestyle=VLM_LINE_STYLE,
            linewidth=VLM_LINE_WIDTH,
            alpha=VLM_ALPHA,
            label=VLM_PERSON2_LABEL)
    
    # Plot RRT trajectories
    ax.plot(rrt_df['time'], rrt_df['rrt_dist_person1'],
            color=RRT_PERSON1_COLOR,
            linestyle=RRT_LINE_STYLE,
            linewidth=RRT_LINE_WIDTH,
            alpha=RRT_ALPHA,
            label=RRT_PERSON1_LABEL)
    
    ax.plot(rrt_df['time'], rrt_df['rrt_dist_person2'],
            color=RRT_PERSON2_COLOR,
            linestyle=RRT_LINE_STYLE,
            linewidth=RRT_LINE_WIDTH,
            alpha=RRT_ALPHA,
            label=RRT_PERSON2_LABEL)
    
    # Formatting
    ax.set_title(PLOT_TITLE, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
    
    # Grid
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINE_STYLE)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save plot
    if SAVE_PLOT:
        plt.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"\nPlot saved to {OUTPUT_FILE}")
    
    # Display plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print("\nVLM Planner:")
    print(f"  Person 1 - Min: {vlm_df['vlm_dist_person1'].min():.3f}m, "
          f"Mean: {vlm_df['vlm_dist_person1'].mean():.3f}m, "
          f"Max: {vlm_df['vlm_dist_person1'].max():.3f}m")
    print(f"  Person 2 - Min: {vlm_df['vlm_dist_person2'].min():.3f}m, "
          f"Mean: {vlm_df['vlm_dist_person2'].mean():.3f}m, "
          f"Max: {vlm_df['vlm_dist_person2'].max():.3f}m")
    
    print("\nRRT Planner:")
    print(f"  Person 1 - Min: {rrt_df['rrt_dist_person1'].min():.3f}m, "
          f"Mean: {rrt_df['rrt_dist_person1'].mean():.3f}m, "
          f"Max: {rrt_df['rrt_dist_person1'].max():.3f}m")
    print(f"  Person 2 - Min: {rrt_df['rrt_dist_person2'].min():.3f}m, "
          f"Mean: {rrt_df['rrt_dist_person2'].mean():.3f}m, "
          f"Max: {rrt_df['rrt_dist_person2'].max():.3f}m")
    print("="*60)


if __name__ == "__main__":
    main()
