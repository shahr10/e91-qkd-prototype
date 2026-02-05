"""
Dependency checker for E91 QKD simulation backends.

Detects installed packages and provides user-friendly install commands.
"""
import importlib.util
from typing import Dict, List, Tuple


def check_backend_dependencies() -> Dict[str, bool]:
    """
    Check which backend dependencies are installed.

    Returns:
        Dictionary with availability status for each dependency
    """
    dependencies = {
        'qiskit': importlib.util.find_spec('qiskit') is not None,
        'qiskit_aer': importlib.util.find_spec('qiskit_aer') is not None,
        'qutip': importlib.util.find_spec('qutip') is not None
    }
    return dependencies


def get_install_commands() -> Dict[str, Dict[str, str]]:
    """
    Get installation commands for missing dependencies.

    Returns:
        Dictionary with install commands for pip and conda
    """
    return {
        'qiskit': {
            'pip': "python -m pip install 'qiskit>=1.0' 'qiskit-aer>=0.14'",
            'conda': "conda install -c conda-forge qiskit qiskit-aer"
        },
        'qutip': {
            'pip': "python -m pip install 'qutip>=5.0'",
            'conda': "conda install -c conda-forge qutip"
        },
        'both': {
            'pip': "python -m pip install 'qiskit>=1.0' 'qiskit-aer>=0.14' 'qutip>=5.0'",
            'conda': "conda install -c conda-forge qiskit qiskit-aer qutip"
        }
    }


def get_missing_dependencies(deps: Dict[str, bool]) -> List[str]:
    """
    Get list of missing dependencies.

    Args:
        deps: Dictionary from check_backend_dependencies()

    Returns:
        List of missing dependency names
    """
    missing = []

    # Qiskit requires both qiskit and qiskit_aer
    if not deps['qiskit'] or not deps['qiskit_aer']:
        missing.append('qiskit')

    if not deps['qutip']:
        missing.append('qutip')

    return missing


def validate_backend_selection(backend: str, deps: Dict[str, bool]) -> Tuple[bool, str]:
    """
    Validate that selected backend is available.

    Args:
        backend: Selected backend ('qiskit' or 'qutip')
        deps: Dictionary from check_backend_dependencies()

    Returns:
        (is_valid, error_message) tuple
    """
    if backend == 'qiskit':
        if not deps['qiskit']:
            return False, "Qiskit is not installed. Please install it to use the Qiskit backend."
        if not deps['qiskit_aer']:
            return False, "Qiskit Aer is not installed. Please install it to use the Qiskit backend."
        return True, ""

    elif backend == 'qutip':
        if not deps['qutip']:
            return False, "QuTiP is not installed. Please install it to use the QuTiP backend."
        return True, ""

    else:
        return False, f"Unknown backend: {backend}"


def get_available_backends(deps: Dict[str, bool]) -> List[str]:
    """
    Get list of available backends based on installed dependencies.

    Args:
        deps: Dictionary from check_backend_dependencies()

    Returns:
        List of available backend names
    """
    available = []

    if deps['qiskit'] and deps['qiskit_aer']:
        available.append('qiskit')

    if deps['qutip']:
        available.append('qutip')

    return available
