from pathlib import Path
from Agents.fund_loader import load_all_fund_positions

if __name__ == "__main__":
    # Load all parsed fund data from your XML files
    df = load_all_fund_positions(data_dir=Path("knowledge_base/Funddata"))

    # Print the top rows
    print(df.head(10))

    # Print column names
    print("Columns:", df.columns.tolist())

    # Print summary
    print(f"Found {df['fund_name'].nunique()} funds and {len(df)} holdings.")
