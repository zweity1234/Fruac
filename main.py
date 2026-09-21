import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import os
import json
import tempfile

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

st.markdown('<div class="app-header"><h1>🍏 Fruac</h1><p>Yapay Zeka Destekli Akıllı Hasat Analizi</p></div>', unsafe_allow_html=True)

# Hafıza Klasörü ve Veri Yönetimi
HAFIZA_KLASOR = "uygulama_hafizasi"
os.makedirs(HAFIZA_KLASOR, exist_ok=True)
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

# Model Yükleme
@st.cache_resource
def load_model():
    return YOLO("best (2).pt")

model = load_model()

# Seçenekler 
URUNLER = {
    "Tüm Meyveler (Elma & Portakal)": {"siniflar": ["apple", "orange"], "gram": 160},
    "Sadece Elma (Apple)": {"siniflar": ["apple"], "gram": 150},
    "Sadece Portakal (Orange)": {"siniflar": ["apple", "orange"], "gram": 200}
}

# --- AYARLAR ---
st.markdown("##### ⚙️ Analiz Seçenekleri")
col1, col2, col3 = st.columns(3)

with col1:
    secilen_etiket = st.selectbox("Sayım Modu", list(URUNLER.keys()))
    kabul_edilen_siniflar = URUNLER[secilen_etiket]["siniflar"]

with col2:
    ortalama_gram = st.number_input(
        "Tane Gramaj (gr)",
        min_value=10,
        max_value=2000,
        value=URUNLER[secilen_etiket]["gram"],
        step=10
    )

with col3:
    guven_esigi = st.slider("Hassasiyet (Confidence)", 0.01, 1.0, 0.15, 0.02)

st.write("")

# --- FOTOĞRAF YÜKLEME ---
st.markdown("##### 📸 Fotoğraf Yükle")
yuklenen_dosyalar = st.file_uploader("Fotoğraf veya Video Yükle", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True)

if yuklenen_dosyalar:
    toplam_adet = 0
    analiz_sonuclari = []
    mevcut_arsiv = arsiv_verilerini_oku()

    with st.spinner("Meyveler tespit ediliyor ve veriler hesaplanıyor..."):
        for dosya in yuklenen_dosyalar:
            if dosya.name.split('.')[-1].lower() in ['mp4', 'mov']:
                st.info(f"🎥 {dosya.name} videosu işleniyor, ağacın etrafı taranıyor... Lütfen bekleyin.")
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(dosya.read())
                
                # VİDEO İŞLEME KISMI - guven_esigi ve bytetrack eklendi!
                sonuclar = model.track(source=tfile.name, conf=guven_esigi, imgsz=640, vid_stride=3, iou=0.6, persist=True, stream=True, tracker="bytetrack.yaml")
                
                for kare_sonucu in sonuclar:
                    if kare_sonucu.boxes is not None and kare_sonucu.boxes.id is not None:
                        for box, obj_id in zip(kare_sonucu.boxes, kare_sonucu.boxes.id):
                            sinif_adi = model.names[int(box.cls)]
                            if sinif_adi in kabul_edilen_siniflar:
                                benzersiz_idler.add(int(obj_id))
                
                toplam_meyve = len(benzersiz_idler)
                hesaplanan_kg = round((toplam_meyve * ortalama_gram) / 1000, 2)
                
                st.success("✅ Ağaç 3D Tarama (Video) Analizi Tamamlandı!")
                st.metric(label="Ağaçtaki Toplam Benzersiz Meyve", value=f"{toplam_meyve} adet")
                st.metric(label="Tahmini Ağaç Verimi", value=f"{hesaplanan_kg} kg")
          
            else:
               
                image = Image.open(dosya).convert("RGB")
                image = ImageOps.exif_transpose(image)
                
                # FOTOĞRAF İŞLEME KISMI - guven_esigi eklendi!
                results = model.predict(image, conf=guven_esigi, imgsz=1024, iou=0.6)
                
                eslesen_kutular = []
                for box in results[0].boxes:
                    sinif_adi = model.names[int(box.cls)]
                    if sinif_adi in kabul_edilen_siniflar:
                        eslesen_kutular.append(box)
                
                adet = len(eslesen_kutular)
                toplam_adet += adet
                hesaplanan_kg = round((adet * ortalama_gram) / 1000, 2)
                
                # Temiz fotoğrafı kaydet
                image.save(os.path.join(HAFIZA_KLASOR, dosya.name))
                
                # Verileri yaz
                mevcut_arsiv[dosya.name] = {
                    "tur": secilen_etiket.split(" ")[0],
                    "adet": adet,
                    "kg": hesaplanan_kg
                }

            cizili_resim = results[0].plot(labels=False)
            analiz_sonuclari.append({
                "dosya_adi": dosya.name,
                "adet": adet,
                "resim": cizili_resim
            })

    arsiv_verisi_kaydet(mevcut_arsiv)
    toplam_kg = (toplam_adet * ortalama_gram) / 1000

    # Rapor Kartları
    st.markdown("---")
    st.markdown("##### 📊 Anlık Analiz Raporu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Analiz Edilen Fotoğraf", f"{len(yuklenen_dosyalar)} Adet")
    m2.metric("Toplam Sayılan Meyve", f"{toplam_adet} Adet")
    m3.metric("Tahmini Toplam Hasat", f"{toplam_kg:.2f} kg")

    # Canlı Analiz Görselleri (Kutucuklu)
    st.markdown("---")
    st.markdown("##### 🔍 Tespit Edilen Alanlar")
    sutunlar = st.columns(min(len(analiz_sonuclari), 2))
    for i, sonuc in enumerate(analiz_sonuclari):
        with sutunlar[i % 2]:
            st.markdown(f"**Görsel:** `{sonuc['dosya_adi']}`")
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
            "tur": "Belirtilmedi",
            "adet": 0,
            "kg": 0.0
        })
        resim_yolu = os.path.join(HAFIZA_KLASOR, dosya_adi)
        
        with kart_sutunlari[i % 2]:
            with st.container(border=True):
                sol_resim, sag_veri = st.columns([1, 2.2])
                with sol_resim:
                    temiz_resim = Image.open(resim_yolu)
                    st.image(temiz_resim, use_container_width=True)
                with sag_veri:
                    st.markdown(f"#### 🍎 {bilgi['tur']}")
                    st.caption(f"Dosya: `{dosya_adi}`")
                    st.markdown(f"**Sayılan:** {bilgi['adet']} Adet")
                    st.markdown(f"**Tahmini Verim:** {bilgi['kg']} kg")

    st.write("")
    if st.button("🗑️ Arşivi Temizle"):
        for dosya in os.listdir(HAFIZA_KLASOR):
            os.remove(os.path.join(HAFIZA_KLASOR, dosya))
        st.rerun()
else:
    st.info("Arşivde henüz kayıtlı bir analiz bulunmuyor.")