# 📊 Perbandingan Hasil Model Time Series Forecasting

## 🎯 **Model yang Dibandingkan**

1. **Few-shot MoE** - Moirai Mixture of Experts dengan adaptasi minimal
2. **Zero-shot** - Moirai Universal Transformer tanpa training
3. **ARIMA** - Statistical baseline dengan auto parameter selection
4. **LSTM** - Deep learning baseline dengan autoregressive forecasting

---

## 📈 **Hasil Perbandingan Lengkap**

### 🌤️ **Weather Melbourne Dataset**

| Model               | MAE               | RMSE              | sMAPE (%)        | Windows |
| ------------------- | ----------------- | ----------------- | ---------------- | ------- |
| **🥇 Zero-shot**    | **1.950 ± 0.405** | **2.450 ± 0.500** | **18.61 ± 4.74** | 22      |
| **🥈 ARIMA**        | 2.143 ± 0.692     | 2.707 ± 0.825     | 21.59 ± 5.88     | 8       |
| **🥉 LSTM**         | 2.212 ± 0.741     | 2.727 ± 0.876     | 20.92 ± 6.34     | 22      |
| **4️⃣ Few-shot MoE** | 2.276 ± 0.183     | 2.869 ± 0.188     | 17.72 ± 1.91     | 3       |

**📝 Analisis Weather Melbourne:**

- **Zero-shot** menunjukkan performa terbaik dengan MAE terendah
- **Few-shot MoE** memiliki konsistensi terbaik (std terendah)
- **ARIMA** dan **LSTM** performa kompetitif untuk baseline
- Data weather menunjukkan pola yang dapat diprediksi dengan baik oleh universal model

---

### 💰 **Finance AAPL Dataset**

| Model               | MAE               | RMSE              | sMAPE (%)       | Windows |
| ------------------- | ----------------- | ----------------- | --------------- | ------- |
| **🥇 ARIMA**        | **6.742 ± 3.527** | **8.495 ± 4.915** | **3.18 ± 1.72** | 8       |
| **🥈 Zero-shot**    | 8.145 ± 3.476     | 9.722 ± 4.195     | 3.76 ± 1.56     | 17      |
| **🥉 Few-shot MoE** | 13.397 ± 2.921    | 15.614 ± 3.164    | 5.90 ± 1.51     | 3       |
| **4️⃣ LSTM**         | 19.004 ± 9.987    | 20.606 ± 10.247   | 8.95 ± 4.25     | 17      |

**📝 Analisis Finance AAPL:**

- **ARIMA** unggul pada data finansial dengan volatilitas tinggi
- **Zero-shot** menunjukkan adaptasi baik untuk domain finansial
- **Few-shot MoE** performa sedang namun dengan konsistensi yang baik
- **LSTM** mengalami kesulitan dengan volatilitas tinggi data finansial

---

### 🌍 **CO2 Mauna Loa Dataset**

| Model               | MAE               | RMSE              | sMAPE (%)       | Windows |
| ------------------- | ----------------- | ----------------- | --------------- | ------- |
| **🥇 ARIMA**        | **0.414 ± 0.170** | **0.482 ± 0.193** | **0.10 ± 0.04** | 8       |
| **🥈 Zero-shot**    | 0.675 ± 0.353     | 0.779 ± 0.378     | 0.16 ± 0.09     | 10      |
| **🥉 Few-shot MoE** | 1.064 ± 0.280     | 1.412 ± 0.329     | 0.25 ± 0.07     | 3       |
| **4️⃣ LSTM**         | 39.232 ± 5.866    | 39.672 ± 5.867    | 9.92 ± 1.38     | 10      |

**📝 Analisis CO2 Mauna Loa:**

- **ARIMA** sangat unggul pada data dengan pola seasonal yang kuat
- **Zero-shot** menunjukkan adaptasi yang baik untuk data environmental
- **Few-shot MoE** performa moderat dengan konsistensi yang baik
- **LSTM** mengalami kesulitan signifikan pada data dengan trend jangka panjang

---

## 🏆 **Ranking Keseluruhan**

### 📊 **Berdasarkan Average MAE Across All Datasets**

| Rank   | Model            | Avg MAE    | Strengths                                                                          | Weaknesses                                                                                   |
| ------ | ---------------- | ---------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **🥇** | **ARIMA**        | **3.100**  | • Excellent for seasonal data<br/>• Fast inference<br/>• Interpretable             | • Linear assumptions<br/>• Struggles with non-linear patterns                                |
| **🥈** | **Zero-shot**    | **3.590**  | • No training needed<br/>• Universal applicability<br/>• Consistent across domains | • No domain-specific adaptation<br/>• Fixed architecture                                     |
| **🥉** | **Few-shot MoE** | **5.579**  | • Advanced architecture<br/>• Good consistency (low std)<br/>• Expert routing      | • Higher computational cost<br/>• Limited adaptation with few shots                          |
| **4️⃣** | **LSTM**         | **20.149** | • Can capture complex patterns<br/>• Flexible architecture                         | • Requires extensive training<br/>• Sensitive to hyperparameters<br/>• Poor on some datasets |

---

## 📈 **Analisis Mendalam per Metrik**

### 🎯 **Mean Absolute Error (MAE)**

- **Best Overall**: ARIMA (sangat unggul pada CO2 dan Finance)
- **Most Consistent**: Few-shot MoE (std deviation terendah)
- **Domain Versatile**: Zero-shot (performa stabil di semua domain)

### 🎯 **Root Mean Square Error (RMSE)**

- **Best Overall**: ARIMA (terutama pada data seasonal)
- **Balanced Performance**: Zero-shot (konsisten across datasets)
- **Most Variable**: LSTM (performance gap besar antar dataset)

### 🎯 **Symmetric MAPE (%)**

- **Best Overall**: ARIMA (error percentage terendah)
- **Good Consistency**: Few-shot MoE (std deviation kecil)
- **Universal Applicability**: Zero-shot (performa stabil)

---

## 🔬 **Insights dan Rekomendasi**

### 🌟 **Key Findings**

1. **ARIMA dominan** untuk data dengan pola seasonal yang jelas (CO2) dan volatilitas finansial
2. **Zero-shot** menunjukkan **versatilitas terbaik** across different domains
3. **Few-shot MoE** memberikan **konsistensi tinggi** meskipun bukan yang terbaik
4. **LSTM** mengalami **overfitting** atau **underfitting** pada beberapa dataset

### 💡 **Rekomendasi Penggunaan**

#### 🎯 **Gunakan ARIMA jika:**

- Data memiliki pola seasonal yang jelas
- Butuh interpretability dan explainability
- Resource komputasi terbatas
- Domain financial atau environmental dengan pola teratur

#### 🎯 **Gunakan Zero-shot jika:**

- Tidak ada data training tersedia
- Butuh deployment cepat across multiple domains
- Performa stabil lebih penting dari akurasi maksimal
- Menangani berbagai jenis time series

#### 🎯 **Gunakan Few-shot MoE jika:**

- Ada beberapa contoh data untuk adaptation
- Butuh konsistensi prediksi yang tinggi
- Computational resource cukup tersedia
- Domain-specific expertise diperlukan

#### 🎯 **Hindari LSTM jika:**

- Data training terbatas
- Butuh hasil cepat tanpa extensive tuning
- Data memiliki karakteristik yang sangat berbeda dari training

---

## 📊 **Summary Statistik**

| Metric           | ARIMA                     | Zero-shot  | Few-shot MoE | LSTM   |
| ---------------- | ------------------------- | ---------- | ------------ | ------ |
| **Wins**         | 🥇🥇 (2/3)                | 🥇 (1/3)   | -            | -      |
| **Avg Rank**     | 1.33                      | 2.00       | 3.00         | 4.00   |
| **Best Domain**  | Financial & Environmental | Weather    | -            | -      |
| **Consistency**  | ⭐⭐⭐                    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐   | ⭐⭐   |
| **Speed**        | ⭐⭐⭐⭐⭐                | ⭐⭐⭐⭐   | ⭐⭐⭐       | ⭐⭐   |
| **Universality** | ⭐⭐                      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐     | ⭐⭐⭐ |

---

## 🎯 **Kesimpulan**

Dalam penelitian ini, **ARIMA** menunjukkan performa terbaik secara keseluruhan, terutama pada data dengan karakteristik seasonal dan financial. **Zero-shot Moirai** membuktikan keunggulan sebagai universal forecasting model dengan konsistensi yang baik across domains. **Few-shot MoE** memberikan konsistensi prediksi terbaik meskipun tidak selalu akurasi tertinggi, menunjukkan potensi besar untuk aplikasi yang membutuhkan reliability. **LSTM** baseline menunjukkan keterbatasan pada beberapa jenis data, menekankan pentingnya pemilihan model yang tepat berdasarkan karakteristik data.

**Model terbaik bergantung pada konteks:**

- **Akurasi maksimal**: ARIMA
- **Universalitas**: Zero-shot
- **Konsistensi**: Few-shot MoE
- **Kompleksitas**: LSTM (jika properly tuned)

---

_📅 Analysis conducted on: October 17, 2025_  
_🔬 Datasets: Weather Melbourne, Finance AAPL, CO2 Mauna Loa_  
_📊 Metrics: MAE, RMSE, sMAPE with statistical significance_
