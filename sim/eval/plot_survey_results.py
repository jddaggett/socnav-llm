"""
Plot survey results comparing VLM-based vs RRT*-based robot preferences across 5 social navigation scenarios.

This script maps Robot A/B responses to VLM/RRT* based on scenario assignments and creates 
three histograms showing survey responses for:
1. Which robot do you prefer?
2. Which robot would you feel more comfortable around?
3. Which robot appears to respect the humans' personal space the best?

Scenario Assignments:
- Scenario 1 (Following Person): Robot A = VLM, Robot B = RRT*
- Scenario 2 (Against Traffic): Robot A = VLM, Robot B = RRT*
- Scenario 3 (Navigating Sidewalk): Robot A = RRT*, Robot B = VLM
- Scenario 4 (Narrow Hallway): Robot A = VLM, Robot B = RRT*
- Scenario 5 (Bypassing Group): Robot A = RRT*, Robot B = VLM
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION - Edit these variables to customize the plots
# ============================================================================

# File path
SURVEY_DATA_FILE = "Social Navigation Preference Survey (Responses) - Form Responses 1.csv"

# Scenario labels (in order as they appear in the survey)
SCENARIO_LABELS = [
    "Following Person",
    "Against Traffic",
    "Navigating Sidewalk",
    "Narrow Hallway",
    "Bypassing Group"
]

# Scenario assignments: which physical robot (A or B) corresponds to which planner
# True = Robot A is VLM (so Robot B is RRT*), False = Robot A is RRT* (so Robot B is VLM)
SCENARIO_A_IS_VLM = [
    True,   # Scenario 1: Following Person - A=VLM, B=RRT*
    True,   # Scenario 2: Against Traffic - A=VLM, B=RRT*
    False,  # Scenario 3: Navigating Sidewalk - A=RRT*, B=VLM
    True,   # Scenario 4: Narrow Hallway - A=VLM, B=RRT*
    False,  # Scenario 5: Bypassing Group - A=RRT*, B=VLM
]

# Question labels
QUESTION_1 = "Which robot do you prefer?"
QUESTION_2 = "Which robot would you feel more comfortable around?"
QUESTION_3 = "Which robot appears to respect the humans' personal space the best?"

# Bar colors
VLM_COLOR = "#4472C4"  # Blue
RRT_COLOR = "#FF0000"  # Red

# Bar transparency
BAR_ALPHA = 0.85

# Figure settings
FIGURE_SIZE = (8, 6)  # Width, height in inches for individual plots
SUBPLOT_SPACING = 0.3  # Space between subplots

# Font sizes
MAIN_TITLE_SIZE = 16
SUBPLOT_TITLE_SIZE = 13
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 10

# Grid settings
SHOW_GRID = True
GRID_AXIS = 'y'  # 'x', 'y', or 'both'
GRID_ALPHA = 0.3
GRID_LINE_STYLE = '--'

# Bar width and spacing
BAR_WIDTH = 0.35
X_LABEL_ROTATION = 15  # Degrees to rotate x-axis labels

# Y-axis settings
Y_AXIS_LABEL = "Number of Responses"

# Output settings
SAVE_PLOT = True
OUTPUT_FILE_Q1 = "survey_q1_preference.png"
OUTPUT_FILE_Q2 = "survey_q2_comfort.png"
OUTPUT_FILE_Q3 = "survey_q3_personal_space.png"
OUTPUT_DPI = 300

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def count_responses(df, question_columns):
    """
    Count VLM and RRT* responses for each scenario by mapping Robot A/B responses.
    
    Args:
        df: DataFrame with survey responses
        question_columns: List of column indices for a specific question across scenarios
        
    Returns:
        Dictionary with scenario names as keys and (vlm_count, rrt_count) tuples
    """
    results = {}
    
    for i, scenario in enumerate(SCENARIO_LABELS):
        col_idx = question_columns[i]
        responses = df.iloc[:, col_idx]
        
        robot_a_count = (responses == 'Robot A').sum()
        robot_b_count = (responses == 'Robot B').sum()
        
        # Map Robot A/B to VLM/RRT* based on scenario assignment
        if SCENARIO_A_IS_VLM[i]:
            # Robot A is VLM, Robot B is RRT*
            vlm_count = robot_a_count
            rrt_count = robot_b_count
        else:
            # Robot A is RRT*, Robot B is VLM
            vlm_count = robot_b_count
            rrt_count = robot_a_count
        
        results[scenario] = (vlm_count, rrt_count)
    
    return results


def create_histogram(data, title, main_title):
    """
    Create a grouped bar chart for one question in a standalone figure.
    
    Args:
        data: Dictionary with scenario names and (vlm_count, rrt_count) counts
        title: Title for the plot
        main_title: Main title for the figure
        
    Returns:
        Figure and axis objects
    """
    scenarios = list(data.keys())
    vlm_counts = [data[s][0] for s in scenarios]
    rrt_counts = [data[s][1] for s in scenarios]
    
    x = np.arange(len(scenarios))
    
    # Create new figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Create bars
    bars1 = ax.bar(x - BAR_WIDTH/2, vlm_counts, BAR_WIDTH, 
                   label='VLM-based', color=VLM_COLOR, alpha=BAR_ALPHA)
    bars2 = ax.bar(x + BAR_WIDTH/2, rrt_counts, BAR_WIDTH,
                   label='RRT*-based', color=RRT_COLOR, alpha=BAR_ALPHA)
    
    # Formatting
    fig.suptitle(main_title, fontsize=MAIN_TITLE_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, pad=10)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=TICK_LABEL_SIZE, rotation=X_LABEL_ROTATION, ha='right')
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    
    # Grid
    if SHOW_GRID:
        ax.grid(True, axis=GRID_AXIS, alpha=GRID_ALPHA, linestyle=GRID_LINE_STYLE)
        ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    return fig, ax


def main():
    """Main function to generate survey result histograms."""
    
    # Load data
    print(f"Loading survey data from {SURVEY_DATA_FILE}...")
    df = pd.read_csv(SURVEY_DATA_FILE)
    total_responses = len(df)
    print(f"  Total responses: {total_responses}")
    
    # Column indices for each question across the 5 scenarios
    # Pattern: Timestamp (col 0), then Q1, Q2, Q3 repeating for each scenario
    # So for each scenario i (0-4): Q1 is at 1+i*3, Q2 is at 2+i*3, Q3 is at 3+i*3
    q1_columns = [1 + i*3 for i in range(5)]  # Columns 1, 4, 7, 10, 13
    q2_columns = [2 + i*3 for i in range(5)]  # Columns 2, 5, 8, 11, 14
    q3_columns = [3 + i*3 for i in range(5)]  # Columns 3, 6, 9, 12, 15
    
    # Count responses for each question
    print("\nCounting responses for each question...")
    q1_data = count_responses(df, q1_columns)
    q2_data = count_responses(df, q2_columns)
    q3_data = count_responses(df, q3_columns)
    
    # Create and save individual figures for each question
    main_title = 'Social Navigation Survey Results: VLM-based vs RRT*-based'
    
    # Question 1
    print("\nCreating Question 1 plot...")
    fig1, ax1 = create_histogram(q1_data, QUESTION_1, main_title)
    if SAVE_PLOT:
        fig1.savefig(OUTPUT_FILE_Q1, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"  Saved to {OUTPUT_FILE_Q1}")
    
    # Question 2
    print("Creating Question 2 plot...")
    fig2, ax2 = create_histogram(q2_data, QUESTION_2, main_title)
    if SAVE_PLOT:
        fig2.savefig(OUTPUT_FILE_Q2, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"  Saved to {OUTPUT_FILE_Q2}")
    
    # Question 3
    print("Creating Question 3 plot...")
    fig3, ax3 = create_histogram(q3_data, QUESTION_3, main_title)
    if SAVE_PLOT:
        fig3.savefig(OUTPUT_FILE_Q3, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"  Saved to {OUTPUT_FILE_Q3}")
    
    # Display all plots
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for q_num, (question, data) in enumerate([
        (QUESTION_1, q1_data),
        (QUESTION_2, q2_data),
        (QUESTION_3, q3_data)
    ], 1):
        print(f"\nQuestion {q_num}: {question}")
        print("-" * 80)
        total_vlm = 0
        total_rrt = 0
        for scenario, (vlm_count, rrt_count) in data.items():
            total_vlm += vlm_count
            total_rrt += rrt_count
            vlm_pct = (vlm_count / total_responses) * 100
            rrt_pct = (rrt_count / total_responses) * 100
            print(f"  {scenario:25s} - VLM: {vlm_count:2d} ({vlm_pct:5.1f}%)  |  "
                  f"RRT*: {rrt_count:2d} ({rrt_pct:5.1f}%)")
        
        total_responses_q = total_vlm + total_rrt
        overall_vlm_pct = (total_vlm / total_responses_q) * 100 if total_responses_q > 0 else 0
        overall_rrt_pct = (total_rrt / total_responses_q) * 100 if total_responses_q > 0 else 0
        print(f"  {'OVERALL':25s} - VLM: {total_vlm:2d} ({overall_vlm_pct:5.1f}%)  |  "
              f"RRT*: {total_rrt:2d} ({overall_rrt_pct:5.1f}%)")
    
    print("="*80)


if __name__ == "__main__":
    main()
