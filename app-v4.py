import streamlit as st
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="PT 파일 메타데이터 뷰어", layout="wide")
st.title("📁 GDrive 기반 .pt 메타데이터 시각화")

uploaded_json = st.file_uploader("📥 메타데이터 JSON 파일 업로드 (final_metadata_with_gdrive_ids.json)", type="json")

if uploaded_json:
    data = json.load(uploaded_json)
    df = pd.DataFrame(data)

    if df.empty or 'filename' not in df.columns or 'gdrive_file_id' not in df.columns:
        st.error("❌ 'filename' 또는 'gdrive_file_id' 컬럼이 누락되었습니다.")
        st.stop()

    st.success(f"✅ {len(df)}개 메타데이터 로딩 완료")

    st.sidebar.header("🔍 필터 조건")
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

    if not filtered_df.empty:
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
            selected_row = selected[0]
            filename = selected_row.get('filename')
            file_id = selected_row.get('gdrive_file_id')

            if filename and file_id:
                st.subheader("📁 선택한 .pt 파일 상세 정보")
                st.write("🧾 파일명:", filename)
                st.write("🔑 GDrive File ID:", file_id)

                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

                try:
                    with st.spinner("📥 .pt 파일 다운로드 및 로딩 중..."):
                        response = requests.get(download_url)
                        if response.status_code != 200:
                            st.error("❌ 다운로드 실패 또는 권한 오류")
                        else:
                            pt_data = torch.load(BytesIO(response.content), map_location="cpu")
                            st.success("✅ .pt 파일 로딩 성공!")

                            img_tensor = pt_data.get("image_tensor")
                            if img_tensor is not None:
                                img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                else:
                                    img_np = img_np.astype(np.uint8)
                                st.image(img_np, caption="📸 image_tensor preview", use_column_width=True)
                            else:
                                st.info("ℹ️ image_tensor 없음")

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
                    st.error(f"❌ torch.load 또는 요청 오류: {e}")

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 필터 결과 CSV 다운로드", data=csv_data, file_name="filtered_metadata.csv", mime="text/csv")
