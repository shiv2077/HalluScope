"""
I/O Utilities Module.

This module provides utilities for file I/O operations,
configuration loading, and data serialization.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import json
import os


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    # TODO: Implement config loading
    path = Path(path)
    
    if path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save.
        path: Output file path.
        indent: JSON indentation.
    """
    # TODO: Implement JSON saving
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        Loaded data.
    """
    # TODO: Implement JSON loading
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict, path: Union[str, Path]) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save.
        path: Output file path.
    """
    # TODO: Implement YAML saving
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Project root path.
    """
    # TODO: Implement project root detection
    # Walk up from current file to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent.parent


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Resolve relative path.
    
    Args:
        path: Path to resolve.
        base: Base directory for relative paths.
        
    Returns:
        Resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    
    base = base or get_project_root()
    return (base / path).resolve()
