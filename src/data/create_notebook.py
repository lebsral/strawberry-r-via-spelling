import nbformat as nbf
from pathlib import Path
import os

# Get project root directory
project_root = Path.cwd()

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell
markdown = nbf.v4.new_markdown_cell("""# GPT-2 Token Extraction Validation

This notebook validates the extracted multi-character, letter-based tokens from the GPT-2 vocabulary and creates visualizations of their distribution.""")

# Create code cell with absolute paths
code = nbf.v4.new_code_cell("""import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os

# Get absolute path to project root (two levels up from this notebook)
notebook_path = Path(os.getcwd()).resolve()  # Get absolute path to current directory
project_root = notebook_path.parent  # Go up one level to project root

# Set up visualization style
plt.style.use('default')  # Use default matplotlib style
sns.set_theme(style="whitegrid")  # Add seaborn grid styling

# Load the extracted tokens
tokens_file = project_root / 'data' / 'processed' / 'gpt2_letter_tokens.json'
print(f"Current working directory: {notebook_path}")
print(f"Project root directory: {project_root}")
print(f"Looking for tokens file at: {tokens_file}")

with open(tokens_file) as f:
    data = json.load(f)
    tokens_data = data['tokens']  # Access the 'tokens' list

# Print raw data structure for debugging
print("\\nData structure example:")
print(tokens_data[0] if tokens_data else "No data found")

# Convert to DataFrame
df = pd.DataFrame(tokens_data)

# Calculate basic statistics
print(f"\\nToken Statistics:")
print(f"Total number of tokens: {len(df)}")
print(f"Average token length: {df['token'].str.len().mean():.2f} characters")
print(f"Median token length: {df['token'].str.len().median()} characters")
print(f"Min token length: {df['token'].str.len().min()} characters")
print(f"Max token length: {df['token'].str.len().max()} characters")

# Create length distribution plot
plt.figure(figsize=(12, 6))
token_lengths = df['token'].str.len()
sns.histplot(data=token_lengths, bins=30)
plt.title('Distribution of Token Lengths')
plt.xlabel('Token Length (characters)')
plt.ylabel('Count')

# Save the plot
results_dir = project_root / 'results' / 'token_analysis'
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_dir / 'token_length_distribution.png')
plt.close()

# Sample tokens of different lengths
print("\\nSample tokens by length:")
for length in range(2, 11):  # Show samples for lengths 2-10
    tokens_of_length = df[df['token'].str.len() == length]['token'].sample(min(5, len(df[df['token'].str.len() == length]))).tolist()
    if tokens_of_length:
        print(f"{length} characters: {', '.join(tokens_of_length)}")

# Save analysis results
analysis_results = {
    'total_tokens': len(df),
    'avg_length': float(df['token'].str.len().mean()),
    'median_length': int(df['token'].str.len().median()),
    'min_length': int(df['token'].str.len().min()),
    'max_length': int(df['token'].str.len().max()),
    'length_distribution': df['token'].str.len().value_counts().to_dict()
}

with open(results_dir / 'analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)
""")

# Add cells to notebook
nb['cells'] = [markdown, code]

# Write the notebook
notebook_dir = project_root / 'notebooks'
notebook_dir.mkdir(exist_ok=True)
nbf.write(nb, notebook_dir / 'token_validation.ipynb')
