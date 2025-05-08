"""Script to create template analysis Jupyter notebooks."""

import nbformat as nbf
from pathlib import Path
import sys

def create_notebook(name, description):
    nb = nbf.v4.new_notebook()

    # Standard imports cell
    imports = '''
    import sys
    from pathlib import Path
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Add project root to path
    project_root = Path.cwd().parent
    sys.path.append(str(project_root))

    # Set standard plotting parameters
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

    # Import project modules
    from src.data.data_loader import TemplateDataLoader, BatchConfig, DataStats
    '''

    cells = [
        nbf.v4.new_markdown_cell(f"# {name}\n\n{description}"),
        nbf.v4.new_code_cell(imports.strip()),
        nbf.v4.new_markdown_cell("## Data Loading"),
        nbf.v4.new_code_cell('''# Initialize data loader
data_dir = project_root / "data" / "processed" / "template_variations"
batch_config = BatchConfig(
    batch_size=32,
    max_length=512,
    similar_length_tolerance=50,
    shuffle=True
)

loader = TemplateDataLoader(
    data_dir=data_dir,
    batch_config=batch_config,
    split_ratios=(0.8, 0.1, 0.1)
)

# Get data statistics
stats = loader.get_stats()
print(f"Total examples: {stats.total_examples}")
print(f"Average sequence length: {stats.avg_sequence_length:.2f}")

# Get training batches
train_batches = list(loader.train_batches())
print(f"Number of training batches: {len(train_batches)}")'''),
        nbf.v4.new_markdown_cell("## Analysis"),
        nbf.v4.new_code_cell('''# Plot template category distribution
plt.figure(figsize=(12, 6))
template_dist = pd.Series(stats.template_distribution)
sns.barplot(x=template_dist.index, y=template_dist.values)
plt.xticks(rotation=45)
plt.title('Distribution of Template Categories')
plt.tight_layout()
plt.savefig(project_root / 'results' / 'token_analysis' / 'template_distribution.png')
plt.close()

# Plot sequence length distribution
plt.figure(figsize=(12, 6))
length_dist = pd.Series(stats.length_distribution)
sns.histplot(data=length_dist.index, weights=length_dist.values, bins=30)
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(project_root / 'results' / 'token_analysis' / 'length_distribution.png')
plt.close()''')
    ]

    nb.cells.extend(cells)
    return nb

def create_template_analysis_notebook():
    """Create the template analysis notebook."""
    return create_notebook(
        name="Template Variation Analysis",
        description="""This notebook analyzes the characteristics and distribution of template variations in our dataset.

Key aspects analyzed:
- Template category distribution
- Token separation patterns
- Sequence length distribution
- Template complexity metrics
- Common patterns and variations"""
    )

def create_template_performance_notebook():
    """Create the template performance analysis notebook."""
    return create_notebook(
        name="Template Performance Analysis",
        description="""This notebook analyzes the effectiveness of different template variations on model performance.

Key metrics analyzed:
- Success rate by template category
- Impact of token separation style
- Sequence length vs performance
- Template complexity vs accuracy
- Error pattern analysis"""
    )

if __name__ == '__main__':
    # Create required directories
    notebooks_dir = Path('notebooks')
    results_dir = Path('results/token_analysis')

    for directory in [notebooks_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Create and save notebooks
    notebooks = {
        'template_analysis.ipynb': create_template_analysis_notebook(),
        'template_performance.ipynb': create_template_performance_notebook()
    }

    for filename, notebook in notebooks.items():
        output_path = notebooks_dir / filename
        nbf.write(notebook, output_path)
        print(f"Created notebook: {output_path}")
