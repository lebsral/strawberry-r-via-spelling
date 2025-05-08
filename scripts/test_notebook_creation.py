"""Test script for validating notebook creation."""

import sys
from pathlib import Path
import nbformat as nbf

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.create_analysis_notebooks import create_notebook, create_template_analysis_notebook, create_template_performance_notebook

def test_notebook_structure():
    """Test that created notebooks have the correct structure."""
    nb = create_notebook("Test Notebook", "Test description")

    # Check basic structure
    assert len(nb.cells) > 0, "Notebook should have cells"
    assert nb.cells[0].cell_type == "markdown", "First cell should be markdown"
    assert "# Test Notebook" in nb.cells[0].source, "Title should be in first cell"

    # Check imports
    assert any("import" in cell.source for cell in nb.cells), "Should have imports"
    assert any("matplotlib" in cell.source for cell in nb.cells), "Should have plotting imports"

    # Check sections
    markdown_cells = [cell.source for cell in nb.cells if cell.cell_type == "markdown"]
    assert any("Data Loading" in source for source in markdown_cells), "Should have data loading section"
    assert any("Analysis" in source for source in markdown_cells), "Should have analysis section"

def test_specific_notebooks():
    """Test creation of specific analysis notebooks."""
    # Test template analysis notebook
    analysis_nb = create_template_analysis_notebook()
    assert "Template Variation Analysis" in analysis_nb.cells[0].source
    assert "template category distribution" in analysis_nb.cells[0].source.lower()

    # Test performance notebook
    performance_nb = create_template_performance_notebook()
    assert "Template Performance Analysis" in performance_nb.cells[0].source
    assert "success rate" in performance_nb.cells[0].source.lower()

def test_directory_creation():
    """Test that required directories are created."""
    # Create test directories
    notebooks_dir = Path('notebooks')
    results_dir = Path('results/token_analysis')

    for directory in [notebooks_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        assert directory.exists(), f"Directory {directory} should exist"
        assert directory.is_dir(), f"{directory} should be a directory"

if __name__ == '__main__':
    # Run tests
    print("Running notebook creation tests...")
    test_notebook_structure()
    test_specific_notebooks()
    test_directory_creation()
    print("All tests passed successfully!")
