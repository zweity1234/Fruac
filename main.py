import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
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
    secilen_etiket = st.selectbox("Meyve Türü", list(URUNLER.keys()))
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

# --- ANALİZ MODU SEÇİMİ ---
analiz_modu = st.radio(
    "📌 Analiz Modunu Seçin:",
    ["📸 Tek Fotoğraf Analizi (Hızlı)", "🌳 4 Cephe Ağaç Analizi (360° Kapsamlı)"],
    horizontal=True
)

st.write("")

# --- FOTOĞRAF İŞLEME FONKSİYONU ---
def fotografi_isle(dosya):
    image = Image.open(dosya).convert("RGB")
    image = ImageOps.exif_transpose(image)
    results = model.predict(image, conf=guven_esigi, imgsz=1024, iou=0.6)
    
    eslesen_kutular = []
    for box in results[0].boxes:
        sinif_adi = model.names[int(box.cls)]
        if sinif_adi in kabul_edilen_siniflar:
            eslesen_kutular.append(box)
    
    adet = len(eslesen_kutular)
    cizili_resim = results[0].plot(labels=False)
    return image, cizili_resim, adet

# ==========================================
# 1. MOD: TEK FOTOĞRAF ANALİZİ
# ==========================================
if analiz_modu == "📸 Tek Fotoğraf Analizi (Hızlı)":
    st.markdown("##### 📸 Fotoğraf Yükle")
    yuklenen_dosyalar = st.file_uploader("Fotoğrafları Seçin", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if yuklenen_dosyalar:
        toplam_adet = 0
        analiz_sonuclari = []
        mevcut_arsiv = arsiv_verilerini_oku()

        with st.spinner("Fotoğraflar analiz ediliyor..."):
            for dosya in yuklenen_dosyalar:
                temiz_resim, cizili_resim, adet = fotografi_isle(dosya)
                toplam_adet += adet
                hesaplanan_kg = round((adet * ortalama_gram) / 1000, 2)

                temiz_resim.save(os.path.join(HAFIZA_KLASOR, dosya.name))
                mevcut_arsiv[dosya.name] = {
                    "tur": secilen_etiket.split(" ")[0],
                    "adet": adet,
                    "kg": hesaplanan_kg
                }

                analiz_sonuclari.append({
                    "dosya_adi": dosya.name,
                    "adet": adet,
                    "resim": cizili_resim
                })

        arsiv_verisi_kaydet(mevcut_arsiv)
        toplam_kg = round((toplam_adet * ortalama_gram) / 1000, 2)

        st.markdown("---")
        st.markdown("##### 📊 Anlık Analiz Raporu")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fotoğraf Sayısı", f"{len(yuklenen_dosyalar)} Adet")
        m2.metric("Sayılan Toplam Meyve", f"{toplam_adet} Adet")
        m3.metric("Tahmini Hasat", f"{toplam_kg} kg")

        st.markdown("---")
        st.markdown("##### 🔍 Tespit Edilen Alanlar")
        sutun_sayisi = max(1, min(len(analiz_sonuclari), 2))
        sutunlar = st.columns(sutun_sayisi)
        for i, sonuc in enumerate(analiz_sonuclari):
            with sutunlar[i % sutun_sayisi]:
                st.markdown(f"**Görsel:** `{sonuc['dosya_adi']}`")
                st.caption(f"Sayılan: **{sonuc['adet']} adet**")
                st.image(sonuc["resim"], channels="BGR", use_container_width=True)
        st.write("")
        st.info("💡 **Bilgilendirme:** Bu sonuçlar yapay zeka destekli bir tahmin modeline dayanmaktadır. Işık yansımaları, yaprak örtüsü ve meyvelerin birbirini gizlemesi (occlusion) gibi doğal koşullar nedeniyle sayımlarda küçük hata payları olabilir.")
# ==========================================
# 2. MOD: 4 CEPHE AĞAÇ ANALİZİ
# ==========================================
else:
    st.markdown("##### 🌳 Ağacın 4 Cephesinden Fotoğrafları Yükleyin")
    st.info("💡 Ağacın etrafında 90° aralıklarla (Ön, Sağ, Arka, Sol) çekilmiş 4 fotoğraf yükleyin.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        f_on = st.file_uploader("1. Cephe (Ön)", type=["jpg", "jpeg", "png"], key="on")
        f_sag = st.file_uploader("2. Cephe (Sağ)", type=["jpg", "jpeg", "png"], key="sag")
    with col_b:
        f_arka = st.file_uploader("3. Cephe (Arka)", type=["jpg", "jpeg", "png"], key="arka")
        f_sol = st.file_uploader("4. Cephe (Sol)", type=["jpg", "jpeg", "png"], key="sol")

    cepheler = [("Ön", f_on), ("Sağ", f_sag), ("Arka", f_arka), ("Sol", f_sol)]
    yuklenen_cepheler = [c for c in cepheler if c[1] is not None]

    if len(yuklenen_cepheler) > 0:
        if st.button("🚀 4 Cephe Ağaç Analizini Başlat", type="primary"):
            toplam_sayilan = 0
            cephe_sonuclari = []

            with st.spinner("Tüm cepheler taranıyor ve 360° ağaç verimi hesaplanıyor..."):
                for isim, dosya in yuklenen_cepheler:
                    temiz_resim, cizili_resim, adet = fotografi_isle(dosya)
                    toplam_sayilan += adet
                    cephe_sonuclari.append({
                        "cephe": isim,
                        "adet": adet,
                        "resim": cizili_resim
                    })

            ortalama_cephe = round(toplam_sayilan / len(yuklenen_cepheler), 1)
            # 4 cephe yüklendiyse doğrudan toplamını alıyoruz; eksik yüklendiyse 4 cepheye oranlıyoruz
            tahmini_agac_toplam = round(ortalama_cephe * 4) if len(yuklenen_cepheler) < 4 else toplam_sayilan
            tahmini_agac_kg = round((tahmini_agac_toplam * ortalama_gram) / 1000, 2)

            st.success("✅ 360° Ağaç Hasat Analizi Tamamlandı!")
            st.markdown("---")
            st.markdown("##### 🌳 Ağaç Verim Raporu")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("İncelenen Cephe", f"{len(yuklenen_cepheler)} / 4")
            r2.metric("Görünen Toplam Meyve", f"{toplam_sayilan} Adet")
            r3.metric("Cephe Başına Ortalama", f"{ortalama_cephe} Adet")
            r4.metric("Tahmini Ağaç Verimi", f"{tahmini_agac_kg} kg")

            st.markdown("---")
            st.markdown("##### 🔍 Cephe İnceleme Detayları")
            c_sutunlar = st.columns(len(cephe_sonuclari))
            for i, c_veri in enumerate(cephe_sonuclari):
                with c_sutunlar[i]:
                    st.markdown(f"**Cephe:** `{c_veri['cephe']}`")
                    st.caption(f"Tespit Edilen: **{c_veri['adet']} Adet**")
                    st.image(c_veri["resim"], channels="BGR", use_container_width=True)
            st.write("")
            st.info("💡 **Bilgilendirme:** Bu sonuçlar yapay zeka destekli bir tahmin modeline dayanmaktadır. Işık yansımaları, yaprak örtüsü ve meyvelerin birbirini gizlemesi (occlusion) gibi doğal koşullar nedeniyle sayımlarda küçük hata payları olabilir.")
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