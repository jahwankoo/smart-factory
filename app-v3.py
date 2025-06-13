import streamlit as st
import zipfile
import tempfile
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import time
import requests
from io import BytesIO

st.set_page_config(page_title="PT Metadata Manager", layout="wide")
st.title("📦 PT 메타데이터 ZIP 업로드 및 시각화")

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

        if not all_meta:
            st.warning("⚠️ 로딩된 메타데이터가 없습니다.")
            st.stop()

        st.success(f"✅ 총 {len(all_meta)}개 메타정보 로딩 완료")

        # DataFrame 변환
        df = pd.DataFrame(all_meta)

        # 통계 시각화
        st.subheader("📊 Label별 분포")
        label_counts = df['label'].value_counts()
        st.bar_chart(label_counts)

        st.subheader("🛠 Table ID별 분포")
        table_counts = df['table_id'].value_counts().sort_index()
        st.line_chart(table_counts)

        st.subheader("📈 Table ID x Label 교차표")
        cross = pd.crosstab(df['table_id'], df['label'])
        st.dataframe(cross)

        # 필터링
        st.sidebar.header("🔎 필터 조건")
        table_ids = sorted(df['table_id'].dropna().unique())
        labels = sorted(df['label'].dropna().unique())

        selected_table = st.sidebar.selectbox("Table ID 선택", [None] + list(table_ids))
        selected_label = st.sidebar.selectbox("Label 선택", [None] + list(labels))

        filtered_df = df.copy()
        if selected_table is not None:
            filtered_df = filtered_df[filtered_df['table_id'] == selected_table]
        if selected_label is not None:
            filtered_df = filtered_df[filtered_df['label'] == selected_label]

        st.subheader("📂 필터링 결과")
        st.write(f"🔍 조건에 맞는 파일: {len(filtered_df)}개")
        st.dataframe(filtered_df, use_container_width=True)

        # Google Drive 링크 기반 .pt 파일 불러오기
        st.subheader("🔗 Google Drive 링크 기반 .pt 파일 미리보기")
        selected_row = st.selectbox("🔍 메타정보에서 파일 선택", filtered_df.to_dict("records"))

        def gdrive_to_direct_url(share_url):
            if "/d/" in share_url:
                file_id = share_url.split("/d/")[1].split("/")[0]
            elif "id=" in share_url:
                file_id = share_url.split("id=")[1].split("&")[0]
            else:
                return None
            return f"https://drive.google.com/uc?export=download&id={file_id}"

        gdrive_link = st.text_input("📎 Google Drive 공유 링크 (.pt 파일)")

        if gdrive_link:
            pt_url = gdrive_to_direct_url(gdrive_link)
            try:
                with st.spinner("📥 .pt 파일 로딩 중..."):
                    response = requests.get(pt_url)
                    pt_data = torch.load(BytesIO(response.content), map_location="cpu")

                    # 이미지 미리보기
                    img_tensor = pt_data.get("image_tensor")
                    if img_tensor is not None:
                        img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                        st.image(img_np, caption="📸 image_tensor preview")
                    else:
                        st.info("ℹ️ image_tensor 없음")

                    # 손 시퀀스
                    hand_seq = pt_data.get("hand_sequence")
                    if hand_seq is not None:
                        st.subheader("✋ Hand Sequence Preview")
                        df_hand = pd.DataFrame(hand_seq.numpy())
                        st.line_chart(df_hand.iloc[:, :5])
                    else:
                        st.info("ℹ️ hand_sequence 없음")

                    # 공압 시퀀스
                    pneu_seq = pt_data.get("pneumatic_sequence")
                    if pneu_seq is not None:
                        st.subheader("🔧 Pneumatic Sequence Preview")
                        df_pneu = pd.DataFrame(pneu_seq.numpy())
                        st.line_chart(df_pneu.iloc[:, :5])
                    else:
                        st.info("ℹ️ pneumatic_sequence 없음")

            except Exception as e:
                st.error(f"❌ .pt 파일 로딩 실패: {e}")

        # 다운로드 버튼
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 필터 결과 CSV 다운로드", data=csv_data, file_name="filtered_metadata.csv", mime="text/csv")
