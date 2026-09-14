import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Meyve Sayar", page_icon="🍎", layout="centered")

st.title("🍎 Tarım Meyve Sayım & Verim Analizi")

# Modeli 'nano' (n) yerine 'medium' (m) versiyona yükselttik (Daha zeki ve detaylı tarama)
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt") # 'n' harfi 'm' oldu

model = load_model()

# AYARLAR KISMI (Telefonda gizlenmemesi için ana ekrana, üst tarafa aldık)
st.markdown("### ⚙️ Analiz Ayarları")
col1, col2 = st.columns(2)
with col1:
    meyve_tipi = st.selectbox("Meyve Türü", ["Elma (Apple)", "Portakal (Orange)"])
with col2:
    guven_esigi = st.slider("Yapay Zeka Hassasiyeti", 0.05, 1.0, 0.15, 0.05)
    
hedef_sinif = "apple" if "Elma" in meyve_tipi else "orange"
varsayilan_gram = 150 if hedef_sinif == "apple" else 200

ortalama_gram = st.number_input("Adet Başı Ortalama Gramaj (gr)", min_value=10, max_value=1000, value=varsayilan_gram, step=10)

st.markdown("---")

# Fotoğraf Yükleme Alanı
yuklenen_dosya = st.file_uploader("Ağaç veya dal fotoğrafı yükleyin...", type=["jpg", "jpeg", "png"])

if yuklenen_dosya is not None:
    image = Image.open(yuklenen_dosya)
    
    with st.spinner("Yapay zeka derin analizi yapıyor, lütfen bekleyin..."):
        results = model.predict(image, conf=guven_esigi, imgsz=1280)
        
        # Hedef meyveyi say
        adet = 0
        for box in results[0].boxes:
            sinif_adi = model.names[int(box.cls[0])]
            if sinif_adi == hedef_sinif:
                adet += 1
        
        # Ağırlık hesabı
        toplam_kg = (adet * ortalama_gram) / 1000
        cizili_resim = results[0].plot()

    # Sonuç Panelleri
    c1, c2 = st.columns(2)
    c1.metric("Tespit Edilen Adet", f"{adet} adet")
    c2.metric("Tahmini Verim", f"{toplam_kg:.2f} kg")

    # Çizilmiş Görsel
    st.image(cizili_resim, channels="BGR", use_container_width=True)