import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2

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

st.markdown('<div class="app-header"><h1>🍏 Fruac</h1><p>Yapay Zeka Destekli Meyve Sayım ve Verim Analizi</p></div>', unsafe_allow_html=True)

# --- HAFIZA KLASÖRÜ OLUŞTURMA ---
HAFIZA_KLASOR = "uygulama_hafizasi"
if not os.path.exists(HAFIZA_KLASOR):
    os.makedirs(HAFIZA_KLASOR)

@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

URUNLER = {
    "Elma (Apple)": {"sinif": "apple", "gram": 150},
    "Portakal (Orange)": {"sinif": "orange", "gram": 200},
    "Muz (Banana)": {"sinif": "banana", "gram": 120},
    "Havuç (Carrot)": {"sinif": "carrot", "gram": 80},
    "Brokoli (Broccoli)": {"sinif": "broccoli", "gram": 300},
}

st.markdown("##### ⚙️ Analiz Seçenekleri")
col1, col2, col3 = st.columns(3)
with col1:
    secilen_etiket = st.selectbox("Meyve Türü", list(URUNLER.keys()))
    hedef_sinif = URUNLER[secilen_etiket]["sinif"]
    hedef_id = [idx for idx, name in model.names.items() if name == hedef_sinif][0]
with col2:
    ortalama_gram = st.number_input("Tane Gramaj (gr)", min_value=10, max_value=2000, value=URUNLER[secilen_etiket]["gram"], step=10)
with col3:
    guven_esigi = st.slider("Hassasiyet (Confidence)", 0.05, 1.0, 0.15, 0.05)

st.write("")

st.markdown("##### 📸 Fotoğraf Yükle")
yuklenen_dosyalar = st.file_uploader("Ağaç fotoğraflarını seçin veya kamerayla çekin...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if yuklenen_dosyalar:
    toplam_adet = 0
    analiz_sonuclari = []

    with st.spinner("Meyveler analiz ediliyor..."):
        for dosya in yuklenen_dosyalar:
            image = Image.open(dosya)
            
            results = model.predict(image, conf=guven_esigi, classes=[hedef_id], imgsz=1280)
            adet = len(results[0].boxes)
            toplam_adet += adet
            cizili_resim = results[0].plot(labels=False)
            
            # --- FOTOĞRAFI HAFIZAYA KAYDETME (RGB'ye Çevirerek) ---
            resim_rgb = cv2.cvtColor(cizili_resim, cv2.COLOR_BGR2RGB)
            kayit_resmi = Image.fromarray(resim_rgb)
            kayit_resmi.save(os.path.join(HAFIZA_KLASOR, dosya.name))
            # -----------------------------------------------------

            analiz_sonuclari.append({
                "dosya_adi": dosya.name,
                "adet": adet,
                "resim": cizili_resim
            })

    toplam_kg = (toplam_adet * ortalama_gram) / 1000

    st.markdown("---")
    st.markdown("##### 📊 Verim Raporu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Analiz Edilen Fotoğraf", f"{len(yuklenen_dosyalar)} Adet")
    m2.metric("Toplam Sayılan Meyve", f"{toplam_adet} Adet")
    m3.metric("Tahmini Toplam Hasat", f"{toplam_kg:.2f} kg")

    st.markdown("---")
    st.markdown("##### 🔍 Tespit Edilen Alanlar")
    sutunlar = st.columns(min(len(analiz_sonuclari), 2))
    for i, sonuc in enumerate(analiz_sonuclari):
        with sutunlar[i % 2]:
            st.markdown(f"**Görsel {i+1}:** `{sonuc['dosya_adi']}`")
            st.caption(f"Sayılan: **{sonuc['adet']} adet**")
            st.image(sonuc["resim"], channels="BGR", use_container_width=True)

# --- GEÇMİŞ HAFIZAYI (GALERİYİ) GÖSTERME ---
st.markdown("---")
st.markdown("##### 📂 Geçmiş Analizler")
kayitli_dosyalar = os.listdir(HAFIZA_KLASOR)

if len(kayitli_dosyalar) > 0:
    # Fotoğrafları yan yana 4'lü sütunlar halinde göster
    hafiza_sutunlar = st.columns(4)
    for i, dosya_adi in enumerate(kayitli_dosyalar):
        with hafiza_sutunlar[i % 4]:
            acilan_resim = Image.open(os.path.join(HAFIZA_KLASOR, dosya_adi))
            st.image(acilan_resim, use_container_width=True)
            st.caption(f"📁 {dosya_adi}")
    
    # Hafızayı temizleme butonu
    if st.button("🗑️ Hafızayı Temizle"):
        for dosya_adi in kayitli_dosyalar:
            os.remove(os.path.join(HAFIZA_KLASOR, dosya_adi))
        st.rerun() # Sayfayı yenile
else:
    st.info("Hafızada henüz kaydedilmiş bir fotoğraf yok.")
    
