import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from datetime import datetime

# ==========================================
# 1. AYARLAR
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXCEL_PATH = "tahmin_verisi.xlsx"        # Veri dosyanız
CKPT_PATH = "models/best-checkpoint.ckpt" # Model dosyanız

print(f"📡 Çalışma Cihazı: {DEVICE}")

# ==========================================
# 2. MODELİ YÜKLE
# ==========================================
# weights_only=False ile güvenlik hatasını aşıyoruz
best_tft = TemporalFusionTransformer.load_from_checkpoint(
    CKPT_PATH, 
    map_location=DEVICE, 
    weights_only=False
)
best_tft.eval()
print(f"✅ Model yüklendi: {CKPT_PATH}")

# ==========================================
# 3. VERİ HAZIRLIĞI
# ==========================================
print("📥 Veri okunuyor...")
df = pd.read_excel(EXCEL_PATH)

# Kolon isimlerini temizle
df.columns = [c.strip() for c in df.columns]

# time_idx kontrolü
if "time_idx" not in df.columns and "ds" in df.columns:
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")
    df["time_idx"] = df["ds"].rank(method="dense").astype(int) - 1

# Kategorik Verileri Hazırla
df["DEPARTMENT_ID_DESC"] = df["DEPARTMENT_ID_DESC"].astype(str).str.strip().str.upper()
df["SKU_GROUP_ID_DESC"] = df["SKU_GROUP_ID_DESC"].astype(str).str.strip().str.upper()
df["group_id"] = df["DEPARTMENT_ID_DESC"] + "_" + df["SKU_GROUP_ID_DESC"]

# Sayısal Dönüşümler
numeric_cols = ["y", "budget_month", "eurtry_month", "inflation", "edu"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
        if col == "inflation":
             mask_high = df[col] > 100
             df.loc[mask_high, col] = df.loc[mask_high, col] / 100

# Bilinmeyen grupları filtrele
known_groups = set(best_tft.dataset_parameters["categorical_encoders"]["group_id"].classes_)
df_filtered = df[df["group_id"].isin(known_groups)].copy()

if len(df_filtered) == 0:
    raise ValueError("❌ Verideki hiçbir grup model tarafından tanınmıyor!")

# ==========================================
# 4. DATASET OLUŞTURMA
# ==========================================
print("Dataset hazırlanıyor...")
dataset = TimeSeriesDataSet.from_parameters(
    best_tft.dataset_parameters,
    df_filtered,
    predict=False, # Modeli test ettiğimiz için False, geleceği tamamen boş tahmin ettireceksek True
    stop_randomization=True
)

dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0, shuffle=False)

# ==========================================
# 5. TAHMİN ALMA
# ==========================================
print("🔮 Tahmin yapılıyor...")

# return_x=True ile zaman ve grup bilgilerini de alıyoruz
raw_predictions = best_tft.predict(
    dataloader, 
    mode="prediction", 
    return_y=True, 
    return_x=True,
    trainer_kwargs=dict(accelerator="cpu")
)

# ==========================================
# 6. SONUÇLARI İŞLEME VE EŞLEŞTİRME
# ==========================================
print("📊 Sonuçlar derleniyor...")

# Decoder tanımları
group_decoder = best_tft.dataset_parameters["categorical_encoders"]["group_id"]
group_col_index = best_tft.dataset_parameters["static_categoricals"].index("group_id") if "group_id" in best_tft.dataset_parameters["static_categoricals"] else -1

results = []
log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Tensörlerden veriyi çekiyoruz
preds = raw_predictions.output
actuals = raw_predictions.y[0]
time_idxs = raw_predictions.x["decoder_time_idx"]
groups_enc = raw_predictions.x["decoder_cat"]

# Döngü ile her tahmini listeye ekle
for i in range(len(preds)):
    # Her batch için decoder uzunluğu kadar tahmin vardır (örn: 6 ay)
    # Biz hepsini satır satır ekliyoruz
    
    # Grup ismini çöz
    enc_id = groups_enc[i][0].item() # Statik olduğu için 0. eleman yetecektir (genelde)
    # Eğer statik değişken sayısı birden fazlaysa doğru indeksi bulmak gerekir.
    # Group ID genelde son eklenendir veya tek ise 0'dır. Garanti yöntem:
    # Not: Hata alırsanız buradaki indexleme yapısını kontrol edin.
    group_name = group_decoder.inverse_transform(torch.tensor([enc_id]))[0]

    prediction_steps = preds[i].numpy() # Tahmin dizisi (örn: 6 eleman)
    actual_steps = actuals[i].numpy()   # Gerçek değer dizisi
    time_steps = time_idxs[i].numpy()   # Zaman dizisi
    
    for step in range(len(prediction_steps)):
        y_pred = float(prediction_steps[step])
        y_true = float(actual_steps[step])
        t_idx = int(time_steps[step])
        
        # Negatif tahminleri sıfıra çek (Satış vb. eksi olamaz)
        if y_pred < 0: y_pred = 0.0

        # Metrik Hesaplamaları
        mae = abs(y_true - y_pred)
        diff = y_true - y_pred
        
        # Accuracy Hesabı (Sıfıra bölme hatasını önle)
        if y_true == 0:
            if y_pred == 0:
                acc = 100.0 # İkisi de 0 ise %100 başarılı
            else:
                acc = 0.0   # Gerçek 0 ama tahmin var ise %0
        else:
            # 1 - (Hata / Gerçek) formülü
            acc = max(0, (1 - (mae / y_true))) * 100

        results.append({
            "Log_Date": log_time,
            "time_idx": t_idx,
            "group_id": group_name,
            "Gercek_Y": y_true,
            "Tahmin": y_pred,
            "Diff (Gercek-Tahmin)": diff,
            "MAE (Mutlak Hata)": mae,
            "Accuracy (%)": round(acc, 2)
        })

# ==========================================
# 7. EXCEL ÇIKTISI
# ==========================================
final_df = pd.DataFrame(results)

# Varsa group_id'yi tekrar parçalayalım okuması kolay olsun
if final_df["group_id"].str.contains("_").all():
    try:
        final_df[["DEPARTMENT", "SKU_GROUP"]] = final_df["group_id"].str.split("_", n=1, expand=True)
        # Sütun sırasını düzenle
        cols = ["Log_Date", "DEPARTMENT", "SKU_GROUP", "time_idx", "Gercek_Y", "Tahmin", "Diff (Gercek-Tahmin)", "MAE (Mutlak Hata)", "Accuracy (%)"]
        final_df = final_df[cols]
    except:
        pass # Parçalanamazsa olduğu gibi bırak

output_file = "final_tahmin_raporu.xlsx"
final_df.to_excel(output_file, index=False)

print(f"\n🚀 İşlem Tamam! Sonuçlar '{output_file}' dosyasına kaydedildi.")
print(final_df.head(10))