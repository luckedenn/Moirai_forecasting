# BAB 4: PERHITUNGAN MANUAL FEW-SHOT FORECASTING MOIRAI MOE

## 4.2 Perhitungan Manual (Versi Sederhana – Moirai-MoE Few-Shot)

Perhitungan manual berikut bertujuan menunjukkan cara kerja Mixture of Experts (MoE) pada satu window uji (few-shot), serta cara menghitung MAE, RMSE, dan sMAPE dari hasil prediksi model. Untuk kesederhanaan, bobot gating dibuat tetap (p₁=0,6; p₂=0,4) dan prediksi tiap expert diberikan eksplisit agar fokus pada langkah perhitungannya.

### 4.2.1 Setup Window Uji

Berdasarkan konfigurasi standardisasi eksperimen:

- Panjang konteks L = 30 (untuk daily data)
- Horizon prediksi H = 7 (untuk Weather Melbourne)
- Untuk kesederhanaan manual, gunakan L = 2 dan H = 2

**Nilai konteks (terakhir sebelum prediksi):** x = [1,00, 1,20]

**Nilai aktual target (dua langkah ke depan):** y = [1,10, 1,30]

**Tabel 4.1 Data window (sederhana)**

| Komponen  | Nilai        |
| --------- | ------------ |
| Konteks x | [1,00, 1,20] |
| Target y  | [1,10, 1,30] |
| Horizon H | 2            |

_Catatan: Pada praktik di skripsi, model menggunakan 6 window evaluasi (sesuai konfigurasi standardisasi). Di sini satu window saja sudah cukup untuk contoh manual._

### 4.2.2 Prediksi per Expert

Berdasarkan arsitektur Moirai-MoE yang memiliki multiple expert networks. Untuk window ini, hasil prediksi tiap expert (dua langkah ke depan) diberikan sebagai:

**Expert-1:** ŷ⁽¹⁾ = [1,05, 1,15]

**Expert-2:** ŷ⁽²⁾ = [1,00, 1,20]

**Tabel 4.2 Prediksi tiap expert**

| Expert | t+1  | t+2  |
| ------ | ---- | ---- |
| ŷ⁽¹⁾   | 1,05 | 1,15 |
| ŷ⁽²⁾   | 1,00 | 1,20 |

### 4.2.3 Agregasi Campuran (Mixture)

Gunakan bobot gating tetap berdasarkan router network: p₁ = 0,6 dan p₂ = 0,4.

**Prediksi gabungan:**

```
ŷ = p₁ŷ⁽¹⁾ + p₂ŷ⁽²⁾
```

**Hitung per langkah:**

**t+1:** 0,6 × 1,05 + 0,4 × 1,00 = 0,63 + 0,40 = 1,03

**t+2:** 0,6 × 1,15 + 0,4 × 1,20 = 0,69 + 0,48 = 1,17

Sehingga **ŷ = [1,03, 1,17]**.

**Tabel 4.3 Prediksi gabungan MoE**

| Langkah | Aktual y | Prediksi ŷ |
| ------- | -------- | ---------- |
| t+1     | 1,10     | 1,03       |
| t+2     | 1,30     | 1,17       |

### 4.2.4 Perhitungan Metrik Error

#### (a) MAE (Mean Absolute Error)

```
MAE = (1/H) Σᵢ₌₁ᴴ |yᵢ - ŷᵢ|
```

**Hitung selisih absolut:**

```
|y - ŷ| = [|1,10 - 1,03|, |1,30 - 1,17|] = [0,07, 0,13]
```

**⇒ MAE = (0,07 + 0,13)/2 = 0,10**

#### (b) RMSE (Root Mean Square Error)

```
RMSE = √[(1/H) Σᵢ₌₁ᴴ (yᵢ - ŷᵢ)²]
```

**Hitung kuadrat selisih:**

```
(y - ŷ)² = [0,07², 0,13²] = [0,0049, 0,0169]
```

**Rata-rata = (0,0049 + 0,0169)/2 = 0,0109**

**⇒ RMSE = √0,0109 ≈ 0,1044**

#### (c) sMAPE (Symmetric Mean Absolute Percentage Error)

```
sMAPE = (100%/H) Σᵢ₌₁ᴴ |yᵢ - ŷᵢ| / [(|yᵢ| + |ŷᵢ|)/2]
```

**t+1:** 0,07 / [(1,10 + 1,03)/2] = 0,07 / 1,065 = 0,06573

**t+2:** 0,13 / [(1,30 + 1,17)/2] = 0,13 / 1,235 = 0,10530

**Rata-rata = (0,06573 + 0,10530)/2 = 0,08552**

**⇒ sMAPE ≈ 8,55%**

**Tabel 4.4 Ringkasan metrik untuk window contoh**

| Metrik | Nilai |
| ------ | ----- |
| MAE    | 0,10  |
| RMSE   | 0,104 |
| sMAPE  | 8,55% |

### 4.2.5 Interpretasi Hasil

Pada window contoh, kesalahan absolut rata-rata sekitar 0,10 dari skala data, dan sMAPE sekitar 8,55%. Nilai ini menunjukkan prediksi few-shot MoE pada konfigurasi sederhana mampu mendekati nilai aktual pada horizon pendek.

**Perbandingan dengan hasil eksperimen sebenarnya:**

- Pada dataset Weather Melbourne: Few-shot MoE mencapai MAE = 2,060 ± 0,830
- Pada dataset Finance AAPL: Few-shot MoE mencapai MAE = 4,155 ± 1,745
- Pada dataset CO2 Mauna Loa: Few-shot MoE mencapai MAE = 1,842 ± 0,409

Pada skripsi, metrik akhir yang dilaporkan adalah rata-rata ± simpangan baku dari 6 window evaluasi; contoh manual ini menunjukkan cara perhitungan untuk satu window representatif.

### 4.2.6 (Opsional) Ilustrasi Gating dengan Softmax

Jika ingin menunjukkan router yang dinamis (tidak tetap), berikut ilustrasi singkat:

**Ringkas encoder konteks:**

```
h = mean(x) = (1,00 + 1,20)/2 = 1,10
```

**Logit gating dari router network:**

```
ℓ₁ = 0,2h = 0,2 × 1,10 = 0,22
ℓ₂ = 0,1h = 0,1 × 1,10 = 0,11
```

**Softmax untuk bobot expert:**

```
p₁ = e^0,22 / (e^0,22 + e^0,11) ≈ 1,246 / (1,246 + 1,116) ≈ 0,528
p₂ = e^0,11 / (e^0,22 + e^0,11) ≈ 1,116 / (1,246 + 1,116) ≈ 0,472
```

**Prediksi dengan bobot dinamis:**

```
t+1: 0,528 × 1,05 + 0,472 × 1,00 = 0,554 + 0,472 = 1,026
t+2: 0,528 × 1,15 + 0,472 × 1,20 = 0,607 + 0,566 = 1,173
```

Ini menunjukkan bahwa di MoE, bobot expert berasal dari router network (softmax), bukan nilai tetap, sehingga model dapat secara adaptif memilih expert yang paling sesuai untuk setiap konteks input.

### 4.2.7 Kesimpulan Perhitungan Manual

Perhitungan manual ini mendemonstrasikan:

1. **Mekanisme MoE:** Kombinasi linear dari multiple expert predictions dengan bobot dari router network
2. **Few-shot learning:** Model dapat melakukan prediksi dengan minimal fine-tuning pada beberapa window saja
3. **Evaluasi metrik:** MAE, RMSE, dan sMAPE sebagai ukuran akurasi prediksi
4. **Konfigurasi standardisasi:** Konsistensi evaluasi dengan 6 window untuk fair comparison

Hasil menunjukkan Few-shot MoE Moirai mampu memberikan prediksi yang kompetitif, menempati peringkat ke-2 setelah ARIMA dalam overall performance, dengan keunggulan pada adaptabilitas dan konsistensi lintas domain yang berbeda.
