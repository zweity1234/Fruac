import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import os
import json

# page confifuraiton
st.set_page_config(page_title="Fruac", page_icon="🍏", layout="wide")

# memory & storage management 
MEMORY_FOLDER = "app_memory"
os.makedirs(MEMORY_FOLDER, exist_ok=True)
DATA_FILE = os.path.join(MEMORY_FOLDER, "database.json")

def load_archive_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
                return {}
        return {}

def save_archive_data(data):
     with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# load YOLO AI model
@st.cache_resource
def load_ai_model():
     return YOLO("best (2).pt")

model = load_ai_model()

# product dictionary
PRODUCTS = {
     "Tüm Meyveler (Elma & Portakal)": {"classes": ["apple", "orange"], "gram": 160},
     "Sadece Elma (Apple)": {"classes": ["apple"], "gram": 150},
     "Sadece Portakal (Orange)": {"classes": ["apple", "orange"], "gram":200}
}

# side panel & user controls
st.markdown("#### Analiz Seçenekleri")
col1, col2, col3 = st.columns(3)

with col1:
    selected_label = st.selectbox("Meyve Türü", list(PRODUCTS.keys()))
    accepted_classes = PRODUCTS[selected_label]["classes"]

with col2:
     average_gram = st.number_input(
          "Tane Gramaj (gr)",
          min_value=10,
          max_value=2000,
          value=PRODUCTS[selected_label]["gram"],
          step=10
     )

with col3:
     confidence_threshold = st.slider("Hassasiyet (Confidence)", 0.01, 1.0, 0.15, 0.02)

st.write("")

# analysis mode selection
analysis_mode = st.radio(
    "Analiz Modunu Seçin:",
    [
         "Tek Fotoğraf Analizi (Hızlı)",
         "Tek Ağaç 4 Cephe Analizi (360°)",
         "Tarla / Bahçe Hasat Tahmini (Örnekleme Modeli)"
    ],
    horizontal=True
)

st.write("")

# core image processing engine
def process_image(file):
     image = Image.open(file).convert("RGB")
     image = ImageOps.exif_transpose(image)
     results = model.predict(image, conf=confidence_threshold, imgsz=1024, iou=0.6)

     matched_boxes = []
     for box in results[0].boxes:
          class_name = model.names[int(box.cls)]
          if class_name in accepted_classes:
               matched_boxes.append(box)

     count = len(matched_boxes)
     plotted_image = results[0].plot(labels=False)
     return image, plotted_image, count


# mode: single photo analysis

if analysis_mode == "Tek Fotoğraf Analizi (Hızlı)":
    st.markdown("##### Fotoğraf Yükle")
    uploaded_files = st.file_uploader("Fotoğrafları Seçin", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
          total_count = 0
          analysis_results = []
          existing_archive = load_archive_data()

          with st.spinner("Fotoğraflar analiz ediliyor..."):
               for file in uploaded_files:
                    clean_image, plotted_image, count = process_image(file)
                    total_count += count 
                    calculated_kg = round((count * average_gram) / 1000, 2)

                    clean_image.save(os.path.join(MEMORY_FOLDER, file.name))
                    existing_archive[file.name] = {
                         "tur": selected_label.split(" ")[0],
                         "adet": count,
                         "kg": calculated_kg
                    }

                    analysis_results.append({
                         "file_name": file.name,
                         "count": count,
                         "image": plotted_image
                    })

                    save_archive_data(existing_archive)
                    total_kg = round((total_count * average_gram / 1000, 2))

                    st.markdown("---")
                    st.markdown("##### Anlık Analiz Raporu")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fotoğraf Sayısı", f"{len(uploaded_files)} Adet")
                    m2.metric("Sayılan Toplam Meyve", f"{total_count} Adet")
                    m3.metric("Tahmini Hasat", f"{total_kg} kg")

                    st.markdown("---")
                    st.markdown("##### Tespit Edilen Alanlar")
                    column_count = max(1, min(len(analysis_results), 2))
                    columns = st.columns(column_count)
                    for i, result in enumerate(analysis_results): 
                         with columns[i % column_count]:
                            st.markdown(f"**Görsel:** '{result['file_name']}'")
                            st.caption(f"Sayılan: **{result['count']} adet**")
                            st.image(result["image"], channels="BGR", use_container_width=True)

                            st.write("")
                            st.info("** Bilgilendirme:** Bu sonuçlar yapay zeka destekli bir tahmin modeline dayanmaktadır. Işık yansımaları ve yaprak örtüsü gibi koşullardan dolayı küçük hata payları olabilir.")

# mode: single tree 360° analysis

elif analysis_mode == "Tek Ağaç 4 Cephe Analizi (360°)":
    st.markdown("##### Ağacın 4 Cephesinden Fotoğraflar Yükleyin")
    st.info("Ağacın etrafında 90° aralıklarla (Ön, Sağ, Arka, Sol) çekilmiş 4 fotoğraf yükleyin.")

    col_a, col_b = st.columns(2)
    with col_a:
        f_front = st.file_uploader("1. Cephe (Ön)", type=["jpg", "jpeg", "png"], key="single_front")
        f_right = st.file_uploader("2. Cephe (Sağ)", type=["jpg", "jpeg", "png"], key="single_right")
    with col_b:
        f_back = st.file_uploader("3. Cephe (Arka)", type=["jpg", "jpeg", "png"], key="single_back")
        f_left = st.file_uploader("4. Cephe (Sol)", type=["jpg", "jpeg", "png"], key="single_left")

    facades = [("Ön", f_front), ("Sağ", f_right), ("Arka", f_back), ("Sol", f_left)]
    uploaded_facades = [f for f in facades if f[1] is not None]

    if len(uploaded_facades) > 0:
        if st.button("4 Cephe Ağaç Analizini Başlat", type="primary"):
            total_detected = 0
            facade_results = []

            with st.spinner("Tüm cepheler taranıyor ve 360° ağaç verimi hesaplanıyor..."):
                for name, file in uploaded_facades:
                    clean_image, plotted_image, count = process_image(file)
                    total_detected += count
                    facade_results.append({
                        "facade": name,
                        "count": count,
                        "image": plotted_image
                    })

            average_facade = round(total_detected / len(uploaded_facades), 1)
            estimated_tree_total = round(average_facade * 4) if len(uploaded_facades) < 4 else total_detected
            estimated_tree_kg = round((estimated_tree_total * average_gram) / 1000, 2)

            st.success("360° Ağaç Hasat Analizi Tamamlandı!")
            st.markdown("---")
            st.markdown("##### Ağaç Verim Raporu")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("İncelenen Cephe", f"{len(uploaded_facades)} / 4")
            r2.metric("Görünen Toplam Meyve", f"{total_detected} Adet")
            r3.metric("Cephe Başına Ortalama", f"{average_facade} Adet")
            r4.metric("Tahmini Ağaç Verimi", f"{estimated_tree_kg} kg")

            st.markdown("---")
            st.markdown("##### Cephe İnceleme Detayları")
            c_columns = st.columns(len(facade_results))
            for i, c_data in enumerate(facade_results):
                with c_columns[i]:
                    st.markdown(f"**Cephe:** `{c_data['facade']}`")
                    st.caption(f"Tespit Edilen: **{c_data['count']} Adet**")
                    st.image(c_data["image"], channels="BGR", use_container_width=True)

# mode: field yield estimation (agritech)

else:
    st.markdown("##### Tarla Geneli Hasat Rekoltesi Tahmini")
    st.caption("Tarlayı temsil eden örnek ağaçların fotoğraflarını yükleyin, yapay zeka tüm bahçenin hasat miktarını çıkarsın.")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        total_field_trees = st.number_input("Tarladaki Toplam Ağaç Sayısı", min_value=1, max_value=50000, value=150, step=10)
    with col_t2:
        sample_tree_count = st.slider("İncelenecek Örnek Ağaç Sayısı", min_value=2, max_value=5, value=3)

    st.markdown("---")
    st.markdown("##### Örnek Ağaçların 4 Cephe Fotoğraflarını Yükleyin")

    tab_names = [f"🌳 {i+1}. Örnek Ağaç" for i in range(sample_tree_count)]
    tabs = st.tabs(tab_names)

    tree_files = {}
    for i, tab in enumerate(tabs):
        with tab:
            st.caption(f"{i+1}. Örnek Ağacın etrafından çekilen 4 cephe fotoğrafını yükleyin:")
            
            col_t_a, col_t_b = st.columns(2)
            with col_t_a:
                t_front = st.file_uploader("1. Cephe (Ön)", type=["jpg", "jpeg", "png"], key=f"field_front_{i}")
                t_right = st.file_uploader("2. Cephe (Sağ)", type=["jpg", "jpeg", "png"], key=f"field_right_{i}")
            with col_t_b:
                t_back = st.file_uploader("3. Cephe (Arka)", type=["jpg", "jpeg", "png"], key=f"field_back_{i}")
                t_left = st.file_uploader("4. Cephe (Sol)", type=["jpg", "jpeg", "png"], key=f"field_left_{i}")

            # Yüklenen fotoğrafları bir liste haline getiriyoruz
            facades_t = [t_front, t_right, t_back, t_left]
            tree_files[i] = [f for f in facades_t if f is not None]

    st.write("")
    if st.button("Tüm Tarlanın Hasatını Hesapla", type="primary"):
        valid_trees = [files for files in tree_files.values() if files and len(files) > 0]
        
        if len(valid_trees) == 0:
            st.warning("Lütfen en az bir örnek ağaç için fotoğraf yükleyin.")
        else:
            tree_fruit_counts = []
            
            with st.spinner("Tarla istatistiği hesaplanıyor..."):
                for tree_idx, files in tree_files.items():
                    if files:
                        tree_total_count = 0
                        for file in files:
                            _, _, count = process_image(file)
                            tree_total_count += count
                        
                        # Eksik cephe varsa 4'e tamamlama oranı
                        if len(files) < 4:
                            tree_total_count = round((tree_total_count / len(files)) * 4)
                        
                        tree_fruit_counts.append(tree_total_count)

            avg_fruit_per_tree = round(sum(tree_fruit_counts) / len(tree_fruit_counts))
            avg_kg_per_tree = round((avg_fruit_per_tree * average_gram) / 1000, 2)
            
            total_field_fruit = avg_fruit_per_tree * total_field_trees
            total_field_kg = round((total_field_fruit * average_gram) / 1000, 2)
            total_field_ton = round(total_field_kg / 1000, 2)

            st.success("Tarla Hasat Rekolte Analizi Başarıyla Tamamlandı!")
            
            st.markdown("---")
            st.markdown("##### Tarla Hasat Tahmin Raporu")
            
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("İncelenen Örnek Ağaç", f"{len(tree_fruit_counts)} Adet")
            t_col2.metric("Ağaç Başına Ort. Meyve", f"{avg_fruit_per_tree} Adet")
            t_col3.metric("Ağaç Başına Ort. Verim", f"{avg_kg_per_tree} kg")
            t_col4.metric("Toplam Ağaç", f"{total_field_trees} Adet")

            st.markdown("---")
            st.markdown("#### TAHMİNİ TOPLAM TARLA REKOLTESİ")
            b1, b2 = st.columns(2)
            b1.metric("Tahmini Toplam Kilo", f"{total_field_kg:,.2f} kg".replace(",", "."))
            b2.metric("Tahmini Toplam Tonaj", f"🏆 {total_field_ton} TON")


# archive section

st.markdown("---")
st.markdown("##### Hasat Arşivi")

archive_data = load_archive_data()
saved_files = [f for f in os.listdir(MEMORY_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if saved_files:
    card_columns = st.columns(2)
    for i, file_name in enumerate(saved_files):
        info = archive_data.get(file_name, {
            "tur": "Belirtilmedi",
            "adet": 0,
            "kg": 0.0
        })
        image_path = os.path.join(MEMORY_FOLDER, file_name)
        
        with card_columns[i % 2]:
            with st.container(border=True):
                left_img, right_info = st.columns([1, 2.2])
                with left_img:
                    clean_img = Image.open(image_path)
                    st.image(clean_img, use_container_width=True)
                with right_info:
                    st.markdown(f"#### {info['tur']}")
                    st.caption(f"Dosya: `{file_name}`")
                    st.markdown(f"**Sayılan:** {info['adet']} Adet")
                    st.markdown(f"**Tahmini Verim:** {info['kg']} kg")

    st.write("")
    if st.button("🗑️ Arşivi Temizle"):
        for file in os.listdir(MEMORY_FOLDER):
            os.remove(os.path.join(MEMORY_FOLDER, file))
        st.rerun()
else:
    st.info("Arşivde henüz kayıtlı bir analiz bulunmuyor.")