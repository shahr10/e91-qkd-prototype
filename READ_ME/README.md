# E91 QKD Prototype - Complete Setup

## Overview

This directory contains the complete, modularized E91 Quantum Key Distribution prototype with all dependencies organized for easy deployment and debugging.

## What's Included

### Main Application
- **e91_app.py** - Streamlit GUI application (simplified, modular version)

### Core Modules
1. **models.py** - Data structures (ExperimentConfig, ExperimentResults, etc.)
2. **quantum_protocol.py** - E91 Protocol implementation
3. **quantum_math.py** - Mathematical functions (entropy, CHSH, key rates)
4. **config.py** - Configuration constants

### Support Modules
5. **analysis.py** - Parametric sweep analysis
6. **visualization.py** - Results plotting
7. **encryption.py** - One-time pad encryption
8. **communication.py** - UDP networking
9. **utils.py** - Helper functions and utilities
10. **styles.py** - CSS styling for GUI
11. **presets.py** - Configuration presets

## Quick Start

### 1. Install Dependencies

**Recommended** (installs both backends):
```bash
pip install -r requirements.txt
```

Or install backends separately:

**Option A** - Using pip:
```bash
# Upgrade pip first
python -m pip install -U pip

# Install both backends (recommended)
python -m pip install 'qiskit>=1.0' 'qiskit-aer>=0.14' 'qutip>=5.0'

# Or install just Qiskit
python -m pip install 'qiskit>=1.0' 'qiskit-aer>=0.14'

# Or install just QuTiP
python -m pip install 'qutip>=5.0'
```

**Option B** - Using conda:
```bash
# Both backends
conda install -c conda-forge qiskit qiskit-aer qutip

# Or just one
conda install -c conda-forge qiskit qiskit-aer
# OR
conda install -c conda-forge qutip
```

### 2. Run the Application

```bash
streamlit run e91_app.py
```

The app will open in your default web browser at http://localhost:8501

If dependencies are missing, the app will show install commands in a friendly UI panel.

## System Requirements

### Required
- Python 3.8 or higher
- NumPy
- Matplotlib
- Streamlit

### Quantum Backend (REQUIRED - at least one)
- **Qiskit >= 1.0** with **Qiskit Aer >= 0.14** (circuit-based simulation)
- **QuTiP >= 5.0** (density matrix simulation)
- **Recommended**: Install BOTH for cross-validation and flexibility

## Module Architecture

```
e91_app.py (Main GUI)
    ├── models.py (Data structures)
    ├── quantum_protocol.py
    │   ├── models.py
    │   ├── utils.py
    │   ├── quantum_math.py
    │   └── qiskit/qutip
    ├── visualization.py
    │   ├── models.py
    │   └── config.py
    ├── analysis.py
    │   ├── models.py
    │   └── quantum_protocol.py
    ├── encryption.py
    │   └── models.py
    ├── communication.py
    ├── utils.py
    │   └── models.py
    ├── styles.py
    └── presets.py
        └── models.py
```

## Features

### 1. E91 Protocol Simulation
- Bell state preparation (Φ+, Φ-, Ψ+, Ψ-)
- Quantum measurements with configurable angles
- CHSH inequality testing for entanglement verification
- QBER (Quantum Bit Error Rate) estimation
- Secret key extraction with finite-size effects

### 2. Physical Models
- **Noise Models**: Depolarizing noise, visibility reduction, intrinsic errors
- **Channel Losses**: Fiber attenuation, detector efficiency, insertion loss
- **Detector Effects**: Dark counts, timing jitter, deadtime, afterpulsing
- **Advanced**: Satellite links, quantum repeaters, multi-pair emission

### 3. Analysis Tools
- Parametric sweeps across any parameter
- Real-time visualization of results
- CHSH vs QBER analysis
- Key rate calculations (asymptotic and finite-size)

### 4. Security Features
- One-time pad encryption with quantum keys
- Message encryption/decryption testing
- Two-party key distribution
- Network communication (UDP-based)

### 5. User Interface
- Modern dark theme
- Interactive parameter controls
- Real-time results visualization
- Experiment history tracking
- Configuration presets

## Usage Examples

### Basic Experiment

1. **Choose a Preset**: Select "Realistic Lab" from the preset dropdown
2. **Set Parameters**: Adjust number of pairs (default: 10,000)
3. **Run Simulation**: Click "Run Experiment"
4. **View Results**: Check QBER, CHSH S, and key generation metrics

### Parametric Sweep

1. Navigate to "Parametric Sweep" tab
2. Select parameter to sweep (e.g., "depolarizing_alpha")
3. Set start value (e.g., 0.01) and end value (e.g., 0.15)
4. Choose number of points (e.g., 10)
5. Click "Run Sweep" and view analysis plots

### Message Encryption

1. Run an experiment first to generate a quantum key
2. Navigate to "Message Test" tab
3. Enter your message
4. Click "Encrypt & Decrypt" to see one-time pad encryption

## Configuration Presets

- **Ideal (No Losses)**: Perfect conditions, no noise or losses
- **Low Noise (α=0.06)**: Minimal depolarizing noise
- **Moderate Fiber (10km)**: 10km fiber link with realistic losses
- **Long Distance (50km)**: 50km fiber link, higher losses
- **Realistic Lab**: Laboratory conditions with multiple noise sources
- **Satellite LEO**: Low Earth Orbit free-space link
- **Custom**: Full manual control

## Testing Individual Modules

You can test modules independently:

```python
# Test quantum protocol
from models import ExperimentConfig
from quantum_protocol import E91Protocol

config = ExperimentConfig(num_pairs=1000, backend="qiskit")
protocol = E91Protocol(config)
results = protocol.run()
print(f"QBER: {results.qber:.4f}, CHSH: {results.chsh_S:.4f}")
```

```python
# Test encryption
from encryption import test_self_message

key = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 10  # 100 bits
msg_test = test_self_message("Hello QKD!", key)
print(f"Success: {msg_test.success}")
print(f"Decrypted: {msg_test.decrypted_message}")
```

```python
# Test visualization
from models import ExperimentConfig, ExperimentResults
from visualization import create_results_plots

# Run experiment first, then:
fig = create_results_plots(config, results)
fig.savefig("results.png")
```

## Troubleshooting

### Import Errors
If you get import errors, make sure all module files are in the same directory as e91_app.py, or the directory is in your Python path.

### Qiskit Not Found
The quantum protocol will try to use Qiskit first. If not available, it will fall back to QuTiP. If neither is available, install one:

```bash
pip install qiskit qiskit-aer
# OR
pip install qutip
```

### Streamlit Issues
Make sure Streamlit is installed and up to date:

```bash
pip install --upgrade streamlit
```

### Performance
For faster simulations:
- Use "qutip" backend instead of "qiskit" for small experiments
- Reduce number of pairs for quick tests
- Disable detailed plots during sweeps

## File Descriptions

### models.py (~300 lines)
Defines all data structures:
- `ExperimentConfig`: 50+ configuration parameters
- `ExperimentResults`: Comprehensive results storage
- `MessageTest`: Message encryption test results
- `MessageListener`: Background UDP listener

### quantum_protocol.py (~700 lines)
Core E91 implementation:
- `E91Protocol` class with full protocol simulation
- Supports Qiskit and QuTiP backends
- Comprehensive loss and noise models
- CHSH calculation and key extraction

### quantum_math.py (~600 lines)
Mathematical functions:
- Information theory (entropy, mutual information)
- Bell inequalities (CHSH computation)
- Key rates (asymptotic and finite-size)
- Statistical bounds (Hoeffding, Chernoff)

### config.py (~1000 lines)
Configuration constants:
- Quantum constants (CHSH bounds, QBER threshold)
- Default measurement angles
- Parameter ranges and validation
- Helper functions (dB conversions)

### visualization.py (~280 lines)
Results plotting:
- 9-subplot comprehensive figure
- Bell test, QBER, pipeline charts
- Correlators heatmap
- Angle visualization
- Security assessment

### analysis.py (~160 lines)
Parametric analysis:
- Parameter sweep execution
- 5 analysis plots
- CHSH vs QBER
- Key rates vs parameters

### encryption.py (~210 lines)
Message encryption:
- Bit conversion functions
- One-time pad XOR encryption
- Self-test and two-party modes
- Network message decryption

### communication.py (~80 lines)
Network functions:
- UDP message sending
- UDP message receiving
- Simple, lightweight communication

### utils.py (~350 lines)
Helper functions:
- Probability normalization
- Satellite loss calculation
- Runtime estimation
- Configuration validation
- Angle and loss previews
- 79 tooltip help texts

### styles.py (~100 lines)
CSS styling:
- Professional dark theme
- Modern UI components
- Responsive layout
- Clean typography

### presets.py (~150 lines)
Configuration presets:
- 6 predefined configurations
- Automatic parameter setting
- Common use cases

## Advanced Usage

### Custom Backends

You can extend the quantum backends by modifying quantum_protocol.py:

```python
# Add custom backend
if config.backend == "custom":
    # Your custom implementation
    pass
```

### Custom Loss Models

Add custom loss models in quantum_protocol.py's `_apply_losses()` method:

```python
# Custom loss
if config.enable_custom_loss:
    custom_efficiency = compute_custom_loss(config)
    valid &= self.rng.random(n) < custom_efficiency
```

### Network Security

For production use, consider:
- Using TCP instead of UDP for reliability
- Adding authentication and encryption
- Implementing proper key management
- Using secure protocols (TLS)

## Performance Optimization

### For Large Simulations
- Use QuTiP backend (faster for >10,000 pairs)
- Enable multiprocessing for sweeps
- Disable real-time plots
- Use HDF5 for result storage

### Memory Usage
- Process results in batches
- Clear history periodically
- Use generators for sweeps
- Compress stored data

## Contributing

To add new features:

1. **New Module**: Create in same directory, add to imports
2. **New Analysis**: Extend analysis.py
3. **New Visualization**: Extend visualization.py
4. **New Preset**: Add to presets.py

## License

MIT License - See original project documentation

## Support

For issues or questions:
1. Check this README
2. Review MODULAR_STRUCTURE.md
3. Test individual modules
4. Check import dependencies

## Version History

- **v7.0 Modular**: Modularized architecture, improved debugging
- **v7.0**: Professional UI, presets, enhanced communication
- **Earlier versions**: See git history

## Authors

Tyler Barr - QKD Research Team

## Acknowledgments

- E91 Protocol: Artur Ekert (1991)
- CHSH Inequality: Clauser, Horne, Shimony, Holt (1969)
- Qiskit: IBM Quantum Team
- QuTiP: QuTiP Development Team
