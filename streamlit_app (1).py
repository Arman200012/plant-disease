import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

SIZE = 160
THRESHOLD = 0.5  # زیر این مقدار، هشدار «مطمئن نیستم» نشان داده می‌شود

st.set_page_config(page_title="تشخیص بیماری برگ گیاه", page_icon="🌿")


@st.cache_resource
def load_everything():
    model = tf.keras.models.load_model("plant_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_everything()
pre = tf.keras.applications.mobilenet_v2.preprocess_input

st.title("🌿 تشخیص بیماری برگ گیاه")
st.write(
    "مدل MobileNetV2 آموزش‌دیده روی دیتاست PlantVillage (۳۸ کلاس). "
    "دقت روی داده‌ی آزمون ۹۳٫۵٪ است، ولی روی عکس‌های مزرعه با پس‌زمینه‌ی شلوغ "
    "ممکن است کمتر باشد."
)

uploaded = st.file_uploader("عکس برگ را انتخاب کنید", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="عکس انتخاب‌شده", use_container_width=True)

    x = pre(np.array(img.resize((SIZE, SIZE)), dtype=np.float32))[None]
    p = model.predict(x, verbose=0)[0]
    top = np.argsort(p)[::-1][:3]

    if p[top[0]] < THRESHOLD:
        st.warning("⚠️ مدل مطمئن نیست. عکس را نزدیک‌تر و با پس‌زمینه‌ی ساده بگیرید.")

    st.subheader("نتیجه")
    for i in top:
        st.write(f"**{class_names[i]}**")
        st.progress(float(p[i]), text=f"{p[i] * 100:.1f}%")
