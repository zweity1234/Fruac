import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Fruac | Fruit Accounting",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ÖZEL CSS TASARIM DOKUNUŞLARI ---
st.markdown("""
    <style>
        /* Ana arka plan ve yazı tipleri */
        .main {
            background-color: #0e1117;
        }
        
        /* Başlık stili */
        .app-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 24px;
            border-radius: 18px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
        }
        .app-header h1 {
            font-size: 2.2rem;
            margin-bottom: 6px;
            font-weight: 700;
        }
        .app-header p {
            color: #d1d8e0;
            font-size: 1rem;
            margin: 0;
        }

        /* Metrik kutucukları (Özet Kartları) */
        div[data-testid="stMetric"] {
            background: #1f242d;
            border: 1px solid #2e3642;
            padding: 16px;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        div[data-testid="stMetricLabel"] {
            color: #9aa0a6 !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="stMetricValue"] {
            color: #00d26a !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }

        /* Fotoğraf yükleme kutusu */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #2a5298;
            border-radius: 14px;
            padding: 10px;
            background: #161a23;
        }

        /* Butonlar ve kontroller */
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(90deg, #00b09b, #96c93d);
            color: white;
            font-weight: bold;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# Model yükleme
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# Ürünler ve ortalama gramajlar
URUNLER = {
    "Elma (Apple)": {"sinif": "apple", "gram": 150},
    "Portakal (Orange)": {"sinif": "orange", "gram": 200},
    "Muz (Banana)": {"sinif": "banana", "gram": 120},
    "Havuç (Carrot)": {"sinif": "carrot", "gram": 80},
    "Brokoli (Broccoli)": {"sinif": "broccoli", "gram": 300},
}

# Özel Başlık Kartı
st.markdown("""
    <div class="app-header">
        <h1>🍏 Fruac</h1>
        <p>Yapay Zeka Destekli Akıllı Ağaç Sayım ve Verim Analizi</p>
    </div>
""", unsafe_allow_html=True)

# --- KONTROL PANELİ ---
st.markdown("##### ⚙️ Analiz Seçenekleri")
col1, col2, col3 = st.columns(3)

with col1:
    secilen_urun = st.selectbox("Ürün Çeşidi", list(URUNLER.keys()))
    hedef_sinif = URUNLER[secilen_urun]["sinif"]

with col2:
    varsayilan_gram = URUNLER[secilen_urun]["gram"]
    ortalama_gram = st.number_input(
        "Tane Gramajı (gr)",
        min_value=10,
        max_value=2000,
        value=varsayilan_gram,
        step=10
    )

with col3:
    guven_esigi = st.slider("Hassasiyet (Confidence)", 0.05, 1.0, 0.15, 0.05)

st.write("")

# --- ÇOKLU FOTOĞRAF YÜKLEME ---
st.markdown("##### 📸 Ağaç Görselleri")
yuklenen_dosyalar = st.file_uploader(
    "Aynı ağacın farklı açılardan çekilmiş fotoğraflarını seçin...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if yuklenen_dosyalar:
    toplam_adet = 0
    analiz_sonuclari = []

    with st.spinner("Görüntüler işleniyor..."):
        for dosya in yuklenen_dosyalar:
            image = Image.open(dosya)
            results = model.predict(image, conf=guven_esigi, imgsz=1280)
            
            foto_adet = 0
            for box in results[0].boxes:
                sinif_adi = model.names[int(box.cls[0])]
                if sinif_adi == hedef_sinif:
                    foto_adet += 1
            
            toplam_adet += foto_adet
            cizili_resim = results[0].plot()
            analiz_sonuclari.append({
                "dosya_adi": dosya.name,
                "adet": foto_adet,
                "resim": cizili_resim
            })

    toplam_kg = (toplam_adet * ortalama_gram) / 1000

    st.markdown("---")
    st.markdown("##### 📊 Toplam Ağaç Verimi")
    m1, m2, m3 = st.columns(3)
    m1.metric("Kare Sayısı", f"{len(yuklenen_dosyalar)} Açı")
    m2.metric("Sayılan Meyve", f"{toplam_adet} Adet")
    m3.metric("Tahmini Hasat", f"{toplam_kg:.2f} kg")

    st.markdown("---")
    st.markdown("##### 🔍 Açı Detayları")

    sutunlar = st.columns(min(len(analiz_sonuclari), 2))
    for i, sonuc in enumerate(analiz_sonuclari):
        with sutunlar[i % 2]:
            st.markdown(f"**Açı {i+1}:** `{sonuc['dosya_adi']}`")
            st.caption(f"Tespit: **{sonuc['adet']} adet** | Tahmini: **{(sonuc['adet'] * ortalama_gram)/1000:.2f} kg**")
            st.image(sonuc["resim"], channels="BGR", use_container_width=True)
