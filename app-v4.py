import streamlit as st
import zipfile
import tempfile
import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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

        df = pd.DataFrame(all_meta)

        # 통계 시각화
        st.subheader("📊 Label별 분포")
        st.bar_chart(df['label'].value_counts())

        st.subheader("🛠 Table ID별 분포")
        st.line_chart(df['table_id'].value_counts().sort_index())

        st.subheader("📈 Table ID x Label 교차표")
        st.dataframe(pd.crosstab(df['table_id'], df['label']))

        # 필터링 UI
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

        # AgGrid에서 행 선택
        gb = GridOptionsBuilder.from_dataframe(filtered_df)
        gb.configure_selection('single')
        grid_options = gb.build()
        grid_response = AgGrid(
            filtered_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=300,
            width='100%',
            allow_unsafe_jscode=True,
        )

        selected = grid_response.get('selected_rows', [])
        if isinstance(selected, list) and len(selected) > 0:
            selected_filename = selected[0].get('filename')

            if selected_filename:
                st.subheader("📁 선택한 .pt 파일 상세 정보")
                folder_id = st.text_input("📂 Google Drive 공유 폴더 ID (processed_segments)", "")

                def build_gdrive_download_url(folder_id, filename):
                    return f"https://drive.google.com/uc?export=download&id={folder_id}&filename={filename}"

                if folder_id:
                    gdrive_url = f"https://drive.google.com/uc?export=download&id={folder_id}&confirm=t"
                    try:
                        with st.spinner("📥 .pt 파일 로딩 중..."):
                            response = requests.get(gdrive_url)
                            pt_data = torch.load(BytesIO(response.content), map_location="cpu")

                            # 이미지 텐서 확인 및 시각화
                            img_tensor = pt_data.get("image_tensor")
                            if img_tensor is not None:
                                img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
                                st.write("📊 image_tensor value range:", img_np.min(), "~", img_np.max())
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                else:
                                    img_np = img_np.astype(np.uint8)
                                st.image(img_np, caption="📸 image_tensor preview")
                            else:
                                st.info("ℹ️ image_tensor 없음")

                            # 시퀀스 데이터
                            hand_seq = pt_data.get("hand_sequence")
                            if hand_seq is not None:
                                st.subheader("✋ Hand Sequence")
                                df_hand = pd.DataFrame(hand_seq.numpy())
                                st.line_chart(df_hand.iloc[:, :5])
                            else:
                                st.info("ℹ️ hand_sequence 없음")

                            pneu_seq = pt_data.get("pneumatic_sequence")
                            if pneu_seq is not None:
                                st.subheader("🔧 Pneumatic Sequence")
                                df_pneu = pd.DataFrame(pneu_seq.numpy())
                                st.line_chart(df_pneu.iloc[:, :5])
                            else:
                                st.info("ℹ️ pneumatic_sequence 없음")
                    except Exception as e:
                        st.error(f"❌ .pt 파일 로딩 실패: {e}")

        # CSV 다운로드
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 필터 결과 CSV 다운로드", data=csv_data, file_name="filtered_metadata.csv", mime="text/csv")
