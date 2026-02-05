# E91 QKD Prototype - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

Open a terminal in this directory and run:

```bash
pip install -r requirements.txt
```

**Note**: This will install all required packages. The installation may take a few minutes.

### Step 2: Run the Application

```bash
streamlit run e91_app.py
```

The application will automatically open in your web browser at: http://localhost:8501

### Step 3: Run Your First Experiment

1. **Choose "Realistic Lab"** from the preset dropdown
2. Click **"Run Experiment"**
3. Wait ~5-10 seconds for results
4. View your results:
   - ✅ QBER (should be ~5-8%)
   - ✅ CHSH S (should be ~2.5-2.7, violating classical bound of 2.0)
   - ✅ Generated key bits

That's it! You've successfully simulated quantum key distribution! 🎉

---

## 📚 What to Try Next

### Experiment with Different Presets

Try these presets to see how different conditions affect the protocol:

- **"Ideal (No Losses)"** - Perfect quantum channel (CHSH ≈ 2.828)
- **"Low Noise (α=0.06)"** - Minimal noise (CHSH ≈ 2.7)
- **"Long Distance (50km)"** - See how distance affects key rate
- **"Satellite LEO"** - Space-based QKD simulation

### Run a Parametric Sweep

1. Go to **"Parametric Sweep"** tab
2. Select "depolarizing_alpha" to sweep
3. Set Start: 0.01, End: 0.15, Points: 10
4. Click **"Run Sweep"**
5. See how noise affects CHSH and key rate

### Test Message Encryption

1. Run an experiment first (any preset)
2. Go to **"Message Test"** tab
3. Type a message: "Hello, Quantum World!"
4. Click **"Encrypt & Decrypt"**
5. See your message encrypted with a quantum key!

---

## ⚙️ Customization

### Adjust Number of Pairs

More pairs = better statistics but slower:
- **100-1,000**: Fast testing (~1 second)
- **10,000**: Default, good balance (~10 seconds)
- **100,000**: High accuracy (~60 seconds)
- **1,000,000**: Research grade (~10 minutes)

### Change Measurement Angles

Open the **"Measurement Angles"** expander to customize Alice and Bob's angles. Default angles are optimized for maximum CHSH violation.

### Enable Different Noise Sources

Try different combinations:
- **Depolarizing Noise**: Quantum decoherence
- **Fiber Loss**: Long-distance attenuation
- **Detector Efficiency**: Imperfect photon detection
- **Dark Counts**: Detector noise

---

## 🔍 Understanding Results

### Key Metrics

**QBER (Quantum Bit Error Rate)**
- < 11%: Secure key can be extracted
- > 11%: Insecure, too many errors

**CHSH S Parameter**
- > 2.0: Bell violation! Entanglement verified
- ≤ 2.0: No violation, classical correlation
- Maximum: 2√2 ≈ 2.828 (Tsirelson bound)

**Key Rate**
- **Asymptotic**: Theoretical limit (infinite data)
- **Finite-Key**: Realistic rate (accounts for finite statistics)

**Detection Efficiency**
- Fraction of photon pairs that survive losses
- Higher is better (100% = ideal)

---

## 🐛 Troubleshooting

### "Backend qiskit not available"

Install Qiskit:
```bash
pip install qiskit qiskit-aer
```

Or use QuTiP backend instead:
```bash
pip install qutip
```
Then select "qutip" in the backend dropdown.

### Slow Performance

For faster simulations:
1. Reduce number of pairs (try 1,000)
2. Use "qutip" backend
3. Uncheck "Show Detailed Plots"

### Module Import Errors

Make sure you're in the PrototypeSetup directory:
```bash
cd c:\Users\TBarr\Desktop\QKDCodes\PrototypeSetup
```

All .py files should be in the same directory.

---

## 📖 Learn More

- **README.md**: Complete documentation
- **MODULAR_STRUCTURE.md**: Code architecture
- **config.py**: See all available parameters

---

## 🎯 Typical Use Cases

### Research/Education
1. Demonstrate Bell inequality violation
2. Study effect of noise on security
3. Compare different channel conditions
4. Explore parameter space

### Protocol Design
1. Test different measurement angles
2. Optimize for specific channels
3. Evaluate security vs. key rate tradeoffs
4. Simulate real-world conditions

### System Design
1. Model fiber-based links
2. Simulate satellite QKD
3. Estimate required detector performance
4. Plan repeater placement

---

## 💡 Tips

1. **Start Simple**: Use "Ideal (No Losses)" first to understand basics
2. **Add Complexity**: Gradually enable noise sources
3. **Compare**: Run multiple experiments and compare in History
4. **Sweep Parameters**: Use sweeps to understand relationships
5. **Save Results**: Download experiment data from History

---

## 🎓 Physics Behind the Numbers

### Why CHSH > 2.0 matters

The CHSH parameter tests if your system is truly quantum:
- Classical physics: S ≤ 2.0 (Bell's inequality)
- Quantum physics: S up to 2√2 ≈ 2.828
- If S > 2.0, you've proven quantum entanglement exists!

### Why QBER < 11% matters

Security requires low errors:
- QBER = 0%: Perfect channel (theoretical)
- QBER < 11%: Secure key can be extracted
- QBER ≥ 11%: Eve could know everything, insecure!

### Key Rate Components

Final key = (Sifted bits) × (Secret fraction) × (Efficiency)
- **Sifted**: Only keep matching basis measurements
- **Secret**: After error correction & privacy amplification
- **Efficiency**: Account for detection losses

---

Enjoy exploring quantum key distribution! 🔐✨
