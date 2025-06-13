import streamlit as st
import zipfile
import tempfile
import os
import json

st.title("📦 .zip 메타파일 업로드 및 조회")

uploaded_zip = st.file_uploader("💾 pt_metadata.zip 파일을 업로드하세요", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "pt_metadata.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # JSON 파일 읽기
        all_meta = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(".json"):
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            all_meta.extend(json.load(f))
                    except Exception as e:
                        st.error(f"❌ {file} 오류: {e}")

        st.success(f"✅ 총 {len(all_meta)}개 메타정보 로딩 완료")

        # 조회 UI
        table_ids = sorted(set(m["table_id"] for m in all_meta if m["table_id"] is not None))
        labels = sorted(set(m["label"] for m in all_meta if m["label"] is not None))

        selected_table = st.selectbox("📌 Table ID 필터", [None] + table_ids)
        selected_label = st.selectbox("🏷️ Label 필터", [None] + labels)

        filtered = [
            m for m in all_meta
            if (selected_table is None or m["table_id"] == selected_table) and
               (selected_label is None or m["label"] == selected_label)
        ]

        st.write(f"🔍 조건에 맞는 파일: {len(filtered)}개")
        st.dataframe(filtered, use_container_width=True)
