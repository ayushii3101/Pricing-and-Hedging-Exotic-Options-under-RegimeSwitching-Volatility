"""
Data Utilities
==============

Functions for loading configurations, saving results, and data handling.
"""

import yaml
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to config file
        
    Returns
    -------
    Dict
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict, output_path: str):
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved configuration to {output_path}")


def save_results(results: Dict, output_path: str, format: str = 'json'):
    """
    Save results to file.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    output_path : str
        Output file path
    format : str
        Output format: 'json', 'pickle', or 'csv'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = convert_to_serializable(results)
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == 'csv':
        # Convert to DataFrame if possible
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved results to {output_path}")


def load_results(input_path: str) -> Dict:
    """
    Load results from file.
    
    Parameters
    ----------
    input_path : str
        Input file path
        
    Returns
    -------
    Dict
        Results dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    if input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            results = json.load(f)
    
    elif input_path.suffix in ['.pkl', '.pickle']:
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
    
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
        results = df.to_dict(orient='list')
    
    else:
        raise ValueError(f"Unknown file type: {input_path.suffix}")
    
    logger.info(f"Loaded results from {input_path}")
    return results


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python types for JSON serialization.
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    Any
        Serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def create_output_directory(base_dir: str = 'results', experiment_name: Optional[str] = None) -> Path:
    """
    Create output directory for results.
    
    Parameters
    ----------
    base_dir : str
        Base directory name
    experiment_name : str, optional
        Experiment name for subdirectory
        
    Returns
    -------
    Path
        Output directory path
    """
    output_dir = Path(base_dir)
    
    if experiment_name:
        output_dir = output_dir / experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_dataframe(df: pd.DataFrame, output_path: str):
    """Save pandas DataFrame to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=True)
    logger.info(f"Saved DataFrame to {output_path}")


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """
    Convert results dictionary to pandas DataFrame.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
        
    Returns
    -------
    pd.DataFrame
        DataFrame representation
    """
    # Handle nested dictionaries and arrays
    flattened = {}
    
    for key, value in results.items():
        if isinstance(value, (list, np.ndarray)):
            if len(value) > 0 and isinstance(value[0], (int, float)):
                flattened[key] = value
        elif isinstance(value, (int, float, str)):
            flattened[key] = [value]
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"{key}_{subkey}"] = [subvalue] if not isinstance(subvalue, list) else subvalue
    
    return pd.DataFrame(flattened)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Log file path
    level : int
        Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info("Logging initialized")


def print_results_summary(results: Dict):
    """
    Print formatted summary of results.
    
    Parameters
    ----------
    results : Dict
        Results dictionary
    """
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key:.<50} {value:.6f}")
        elif isinstance(value, str):
            print(f"{key:.<50} {value}")
        elif isinstance(value, (list, np.ndarray)):
            if len(value) <= 5:
                print(f"{key:.<50} {value}")
            else:
                print(f"{key:.<50} [array of length {len(value)}]")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float)):
                    print(f"  {subkey:.<48} {subvalue:.6f}")
    
    print("=" * 70 + "\n")
