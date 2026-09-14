import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import json

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Fruac | Fruit Accounting",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Arayüz Stili
st.markdown("""
    <style>
        .app-header { background: linear-gradient(135deg, #1b4d3e 0%, #2e8b57 100%); padding: 22px; border-radius: 16px; color: white; margin-bottom: 20px; text-align: center; }
        .app-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
        .app-header p { color: #e0f2e9; margin-top: 6px; font-size: 0.95rem; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0; padding: 16px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header"><h1>🍏 Fruac</h1><p>Meyve Sayım ve Hasat Analizi</p></div>', unsafe_allow_html=True)

HAFIZA_KLASOR = "uygulama_hafizasi"
if not os.path.exists(HAFIZA_KLASOR):
    os.makedirs(HAFIZA_KLASOR)

VERI_DOSYASI = os.path.join(HAFIZA_KLASOR, "veriler.json")

def arsiv_verilerini_oku():
    if os.path.exists(VERI_DOSYASI):
        try:
            with open(VERI_DOSYASI, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def arsiv_verisi_kaydet(veriler):
    with open(VERI_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(veriler, f, ensure_ascii=False, indent=4)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

URUNLER = {
    "Elma (Apple)": {"sinif": "apple", "gram": 150},
    "Portakal (Orange)": {"sinif": "orange", "gram": 200}
}

# --- AYARLAR ---
st.markdown("##### ⚙️ Analiz Seçenekleri")
col1, col2, col3 = st.columns(3)

with col1:
    secilen_etiket = st.selectbox("Meyve Türü", list(URUNLER.keys()))
    hedef_sinif = URUNLER[secilen_etiket]["sinif"]

with col2:
    ortalama_gram = st.number_input(
        "Tane Gramaj (gr)",
        min_value=10,
        max_value=2000,
        value=URUNLER[secilen_etiket]["gram"],
        step=10
    )

with col3:
    guven_esigi = st.slider("Hassasiyet (Confidence)", 0.01, 1.0, 0.10, 0.02)

st.write("")

# --- FOTOĞRAF YÜKLEME ---
st.markdown("##### 📸 Fotoğraf Yükle")
yuklenen_dosyalar = st.file_uploader(
    "Ağaç fotoğraflarını seçin...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if yuklenen_dosyalar:
    toplam_adet = 0
    analiz_sonuclari = []
    mevcut_arsiv = arsiv_verilerini_oku()

    with st.spinner("Model meyveleri analiz ediyor..."):
        for dosya in yuklenen_dosyalar:
            image = Image.open(dosya).convert("RGB")
            
            # Sınıf kısıtlaması olmadan tahmin yap
            results = model.predict(image, conf=guven_esigi, imgsz=640)
            
            # Seçilen meyveye ait kutuları filtrele
            hedef_kutular = []
            for box in results[0].boxes:
                sinif_adi = model.names[int(box.cls)]
                if sinif_adi == hedef_sinif:
                    hedef_kutular.append(box)

            adet = len(hedef_kutular)
            toplam_adet += adet
            hesaplanan_kg = round((adet * ortalama_gram) / 1000, 2)

            # Temiz resmi arşive kaydet
            image.save(os.path.join(HAFIZA_KLASOR, dosya.name))

            # Verileri sözlüğe ekle
            mevcut_arsiv[dosya.name] = {
                "meyve": secilen_etiket,
                "adet": adet,
                "kg": hesaplanan_kg
            }

            cizili_resim = results[0].plot(labels=False)
            analiz_sonuclari.append({
                "dosya_adi": dosya.name,
                "adet": adet,
                "resim": cizili_resim,
                "toplam_kutular": len(results[0].boxes)
            })

    arsiv_verisi_kaydet(mevcut_arsiv)
    toplam_kg = (toplam_adet * ortalama_gram) / 1000

    # Rapor
    st.markdown("---")
    st.markdown("##### 📊 Anlık Analiz Raporu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Analiz Edilen Fotoğraf", f"{len(yuklenen_dosyalar)} Adet")
    m2.metric("Toplam Sayılan Meyve", f"{toplam_adet} Adet")
    m3.metric("Tahmini Toplam Hasat", f"{toplam_kg:.2f} kg")

    # Tespit Görselleri
    st.markdown("---")
    st.markdown("##### 🔍 Tespit Edilen Alanlar")
    sutunlar = st.columns(min(len(analiz_sonuclari), 2))
    for i, sonuc in enumerate(analiz_sonuclari):
        with sutunlar[i % 2]:
            st.markdown(f"**Görsel:** `{sonuc['dosya_adi']}`")
            if sonuc["adet"] == 0 and sonuc["toplam_kutular"] > 0:
                st.warning(f"⚠️ Model bir şeyler buldu ama hiçbiri '{secilen_etiket}' sınıfına uymadı.")
            elif sonuc["adet"] == 0:
                st.info("ℹ️ Bu hassasiyet eşiğinde model hiç meyve tespit edemedi. Hassasiyet slider'ını sola çekmeyi deneyin.")
            else:
                st.caption(f"Sayılan: **{sonuc['adet']} adet**")
            st.image(sonuc["resim"], channels="BGR", use_container_width=True)

# --- ARŞİV BÖLÜMÜ ---
st.markdown("---")
st.markdown("##### 🌳 Hasat Arşivi")

arsiv_verileri = arsiv_verilerini_oku()
kayitli_dosyalar = [f for f in os.listdir(HAFIZA_KLASOR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if kayitli_dosyalar:
    kart_sutunlari = st.columns(2)
    for i, dosya_adi in enumerate(kayitli_dosyalar):
        bilgi = arsiv_verileri.get(dosya_adi, {
            "meyve": "Belirtilmedi",
            "adet": 0,
            "kg": 0.0
        })
        resim_yolu = os.path.join(HAFIZA_KLASOR, dosya_adi)
        
        with kart_sutunlari[i % 2]:
            with st.container(border=True):
                sol_resim, sag_veri = st.columns([1, 2])
                with sol_resim:
                    temiz_resim = Image.open(resim_yolu)
                    st.image(temiz_resim, use_container_width=True)
                with sag_veri:
                    st.markdown(f"**{bilgi['meyve']}**")
                    st.caption(f"📁 `{dosya_adi}`")
                    st.markdown(f"🔢 **{bilgi['adet']}** Adet")
                    st.markdown(f"⚖️ **{bilgi['kg']}** kg")

    st.write("")
    if st.button("🗑️ Arşivi Temizle"):
        for dosya in os.listdir(HAFIZA_KLASOR):
            os.remove(os.path.join(HAFIZA_KLASOR, dosya))
        st.rerun()
else:
    st.info("Arşivde henüz kayıtlı bir analiz bulunmuyor.")
