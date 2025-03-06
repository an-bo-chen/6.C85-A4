import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def generate_viz_1(df):
    # Selecting relevant columns and dropping missing values
    df_filtered = df[['mos_ethnicity', 'complainant_ethnicity']].dropna()

    # Grouping data by police officer ethnicity and complainant ethnicity
    grouped = df_filtered.groupby(['mos_ethnicity', 'complainant_ethnicity']).size().unstack(fill_value=0)

    # Reorder ethnicities for consistency
    ethnicities = ['White', 'Hispanic', 'Black', 'Asian', 'American Indian']
    grouped = grouped.reindex(ethnicities, fill_value=0)

    # Compute same-race and other-race complaint counts
    same_race_counts = grouped.apply(lambda row: row.get(row.name, 0), axis=1)
    total_counts = grouped.sum(axis=1)
    other_race_counts = total_counts - same_race_counts

    # Compute percentages
    same_race_perc = (same_race_counts / total_counts * 100).fillna(0)
    other_race_perc = (other_race_counts / total_counts * 100).fillna(0)

    # Plotting the horizontal stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    y_positions = np.arange(len(ethnicities))

    # Create stacked bars
    ax.barh(y_positions, other_race_counts, color='darkred', label='Other Race')
    ax.barh(y_positions, same_race_counts, left=other_race_counts, color='lightgreen', label='Same Race')

    # Adding text labels for counts and percentages
    for i, (other_count, same_count, other_perc, same_perc) in enumerate(
        zip(other_race_counts, same_race_counts, other_race_perc, same_race_perc)
    ):  
        if i == 3: # Skip Asian because text is too crowded
            continue

        if other_count > 0:
            ax.text(
                other_count / 2, i, 
                f"{other_perc:.2f}%\n{int(other_count)}", 
                ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold'
            )
        if same_count > 0:
            ax.text(
                other_count + same_count / 2, i, 
                f"{same_perc:.2f}%\n{int(same_count)}", 
                ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold'
            )

    # Configure y‐axis labels and title
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ethnicities)
    ax.set_ylabel("Race of Police Officer")
    ax.set_xlabel("Number of Misconduct Allegations")
    ax.set_title("Exposing Racial Biases in NYPD Misconduct Allegations")

    # Move legend to upper right
    ax.legend(title="Race of Complainant", loc='lower right')

    ax.invert_yaxis()

def generate_viz_2(df):
    # Define outcome categories based on 'board_disposition'
    def categorize_outcome(disposition):
        if "Substantiated" in disposition:
            return "Substantiated"
        elif "Exonerated" in disposition:
            return "Exonerated"
        else:
            return "Unsubstantiated"

    # Extract relevant columns
    df_filtered = df[['unique_mos_id', 'board_disposition', 'mos_ethnicity']].dropna()

    # Apply outcome classification
    df_filtered['Outcome'] = df_filtered['board_disposition'].apply(categorize_outcome)

    # Aggregate data: count occurrences and calculate percentages
    outcome_counts = df_filtered.groupby(['mos_ethnicity', 'Outcome']).size().unstack(fill_value=0)
    outcome_percentages = outcome_counts.div(outcome_counts.sum(axis=1), axis=0) * 100

    # Sort by total complaints (optional)
    outcome_percentages = outcome_percentages.loc[outcome_percentages.sum(axis=1).sort_values(ascending=False).index]
    outcome_percentages_filtered = outcome_percentages.drop(index='American Indian', errors='ignore')

    # Define colormap for the plot
    colormap = colors.LinearSegmentedColormap.from_list("", ["lightskyblue","skyblue","deepskyblue"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    outcome_percentages_filtered.plot(kind='barh', stacked=True, colormap=colormap, ax=ax)

    # Labels and Title
    plt.xlabel('Percentage of Total Complaints')
    plt.ylabel('Race of Police Officer')
    plt.title('Revealing Equitable NYPD Complaint Outcomes Across Each Police Racial Demographic')

    # Add percentage labels
    for i, (ethnicity, row) in enumerate(outcome_percentages_filtered.iterrows()):
        cumulative = 0
        for outcome, value in row.items():
            plt.text(cumulative + value / 2, i, f"{value:.2f}%", va='center', ha='center', fontsize=10, color='black')
            cumulative += value

    # Move legend to upper right
    plt.legend(title="Outcome", loc="upper right")

    ax.invert_yaxis()


if __name__ == "__main__":
    # Load the dataset
    file_path = "CCRB-Complaint-Data_202007271729/allegations_202007271729.csv"
    df = pd.read_csv(file_path)

    # Call the visualization function
    # generate_viz_1(df)
    # generate_viz_2(df)

    plt.show()

