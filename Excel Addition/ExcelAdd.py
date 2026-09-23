from pathlib import Path
import pandas as pd

# Path to the main directory containing your season folders
root_dir = Path(r"E:\University\Applied Oceanography\Dissertation\oceanography-university\Excel Addition\data")

# List of seasons to process
seasons = ["autumn", "spring", "summer", "winter"]

# Find all CSV files in the directory tree once
all_csv_files = list(root_dir.rglob("*.csv"))

for season in seasons:
    season_dfs = []

    for csv_file in all_csv_files:
        # Ignore any combined master files created on previous runs
        if "_combined.csv" in csv_file.name:
            continue

        # Check if the folder name belongs to the current season (e.g., "autumn_2025-09-21")
        if csv_file.parent.name.lower().startswith(season):
            try:
                df = pd.read_csv(csv_file)

                # Optional metadata columns tracking origin folder/file
                df["Source_Folder"] = csv_file.parent.name
                df["Source_File"] = csv_file.name

                season_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")

    # Merge and save a CSV for the season if files were found
    if season_dfs:
        season_df = pd.concat(season_dfs, ignore_index=True)
        output_path = root_dir / f"{season}_2025_combined.csv"
        season_df.to_csv(output_path, index=False)
        print(f"Done: Created '{output_path.name}' from {len(season_dfs)} files.")
    else:
        print(f"No files matching season '{season}' were found.")