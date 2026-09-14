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

# Arayüz Stili
st.markdown("""
    <style>
        .app-header {
            background: linear-gradient(135deg, #1b4d3e 0%, #2e8b57 100%);
            padding: 22px;
            border-radius: 16px;
            color: white;
            margin-bottom: 20px;
            text-align: center;
        }
        .app-header h1 {
            font-size: 2.2rem;
            margin: 0;
            font-weight: 700;
        }
        .app-header p {
            color: #e0f2e9;
            margin-top: 6px;
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 0.9rem !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #1b4d3e !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("""
    <div class="app-header">
        <h1>🍏 Fruac</h1>
        <p>Yapay Zeka Destekli Meyve Sayım ve Verim Analizi</p>
    </div>
""", unsafe_allow_html=True)

# Modeli Önbelleğe Alarak Yükleme
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# Desteklenen Ürünler ve Varsayılan Gramajlar (COCO Sınıfları)
URUNLER = {
    "Elma (Apple)": {"sinif": "apple", "gram": 150},
    "Portakal (Orange)": {"sinif": "orange", "gram": 200},
    "Muz (Banana)": {"sinif": "banana", "gram": 120},
    "Havuç (Carrot)": {"sinif": "carrot", "gram": 80},
    "Brokoli (Broccoli)": {"sinif": "broccoli", "gram": 300},
}

# --- KONTROL AYARLARI ---
st.markdown("##### ⚙️ Analiz Seçenekleri")
col1, col2, col3 = st.columns(3)

with col1:
    secilen_etiket = st.selectbox("Meyve Türü", list(URUNLER.keys()))
    hedef_sinif = URUNLER[secilen_etiket]["sinif"]
    # Modelin sınıf indeksini bulma
    hedef_id = None
    for idx, name in model.names.items():
        if name == hedef_sinif:
            hedef_id = idx
            break

with col2:
    varsayilan_gram = URUNLER[secilen_etiket]["gram"]
    ortalama_gram = st.number_input(
        "Tane Başı Ortalama Gramaj (gr)",
        min_value=10,
        max_value=2000,
        value=varsayilan_gram,
        step=10
    )

with col3:
    guven_esigi = st.slider("Hassasiyet Eşiği", 0.05, 1.0, 0.15, 0.05)

st.write("")

# --- FOTOĞRAF YÜKLEME ---
st.markdown("##### 📸 Fotoğraf Yükle")
yuklenen_dosyalar = st.file_uploader(
    "Ağaç fotoğraflarını seçin veya kamerayla çekin...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if yuklenen_dosyalar:
    toplam_adet = 0
    analiz_sonuclari = []

    with st.spinner("Meyveler sayılıyor ve analiz ediliyor..."):
        for dosya in yuklenen_dosyalar:
            image = Image.open(dosya)
            
            # Sadece hedef meyveyi ara (classes filtresi)
            results = model.predict(
                image,
                conf=guven_esigi,
                classes=[hedef_id] if hedef_id is not None else None,
                imgsz=1280
            )
            
            adet = len(results[0].boxes)
            toplam_adet += adet
            
            # Görsel üstünde sadece kutular olsun (etiket ve sayılar kapalı)
            cizili_resim = results[0].plot(labels=False)
            
            analiz_sonuclari.append({
                "dosya_adi": dosya.name,
                "adet": adet,
                "resim": cizili_resim
            })

    # Verim Hesabı
    toplam_kg = (toplam_adet * ortalama_gram) / 1000

    # --- ÖZET METRİKLER ---
    st.markdown("---")
    st.markdown("##### 📊 Verim Raporu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Analiz Edilen Fotoğraf", f"{len(yuklenen_dosyalar)} Adet")
    m2.metric("Toplam Sayılan Meyve", f"{toplam_adet} Adet")
    m3.metric("Tahmini Toplam Hasat", f"{toplam_kg:.2f} kg")

    # --- DETAYLI GÖRSELLER ---
    st.markdown("---")
    st.markdown("##### 🔍 Tespit Detayları")
    
    sutunlar = st.columns(min(len(analiz_sonuclari), 2))
    for i, sonuc in enumerate(analiz_sonuclari):
        with sutunlar[i % 2]:
            st.markdown(f"**Görsel {i+1}:** `{sonuc['dosya_adi']}`")
            st.caption(f"Sayılan: **{sonuc['adet']} adet** | Tahmini: **{(sonuc['adet'] * ortalama_gram) / 1000:.2f} kg**")
            st.image(sonuc["resim"], channels="BGR", use_container_width=True)
