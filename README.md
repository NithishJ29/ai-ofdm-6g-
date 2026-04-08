# ai-ofdm-6g-
AI-Enhanced OFDM Receiver with Channel Estimation and LDPC (6G-Oriented)
# AI-Enhanced OFDM Receiver for 6G (NVIDIA Sionna)

This project implements a **next-generation wireless communication system** combining:

- OFDM (Orthogonal Frequency Division Multiplexing)
- Rayleigh fading channel
- LS Channel Estimation
- MMSE Equalization
- 5G LDPC Coding
- AI-based Neural Demapper

📡 Built using NVIDIA Sionna for **6G AI-native PHY research**

---

##  Key Features

- BER & BLER vs SNR analysis
- AI-based demapper (deep learning)
- Classical vs AI receiver comparison
- Realistic channel estimation (LS)
- MMSE equalization with estimation error
- End-to-end PHY simulation

---

## System Architecture

Bits 
  ↓
LDPC Encoder
  ↓
QAM Mapper
  ↓
OFDM Resource Grid
  ↓
Channel (Rayleigh + AWGN)
  ↓
Channel Estimation (LS)
  ↓
MMSE Equalization
  ↓
+-----------------------+
| Classical Demapper    |
| AI Neural Demapper    |
+-----------------------+
  ↓
LDPC Decoder
  ↓
BER / BLER Calculation
---

## 📊 Results

| System | Description |
|------|------------|
| Ideal Classical | Perfect channel knowledge |
| Classical | LS estimation + classical demapper |
| AI Receiver | LS estimation + neural demapper |
| LDPC + AI | Full 5G-compliant system |

---

## 📈 Performance Plot

![BER vs SNR](/ber_vs_snr.png)

---

## 🧪 Tech Stack

- Python
- TensorFlow
- NVIDIA Sionna
- NumPy
- Matplotlib

---

## ▶️ How to Run

### 1. Create environment


python -m venv sionna_env
sionna_env\Scripts\activate   
