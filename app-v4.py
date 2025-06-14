import streamlit as st
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
import gdown  # requests 대신 gdown 사용
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- 페이지 설정 및 초기화 ---
st.set_page_config(page_title="PT 파일 메타데이터 뷰어", layout="wide")

# session_state 초기화
if 'pt_data' not in st.session_state:
    st.session_state.pt_data = None
if 'last_selected_id' not in st.session_state:
    st.session_state.last_selected_id = None

# --- 함수 정의 ---
@st.cache_data # 데이터 로딩 캐싱
def load_json_data(uploaded_file):
    """업로드된 JSON 파일을 데이터프레임으로 로드합니다."""
    try:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        if 'filename' not in df.columns or 'gdrive_file_id' not in df.columns:
            st.error("❌ 'filename' 또는 'gdrive_file_id' 컬럼이 누락되었습니다.")
            return None
        return df
    except Exception as e:
        st.error(f"❌ JSON 파일 처리 중 오류 발생: {e}")
        return None

@st.cache_data # PT 파일 다운로드 및 로딩 캐싱
def download_and_load_pt(file_id):
    """gdown을 사용하여 PT 파일을 다운로드하고 로드합니다."""
    if not file_id:
        return None
    
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        with st.spinner(f"📥 .pt 파일 다운로드 및 로딩 중... (ID: {file_id})"):
            # gdown.download의 output='-' 옵션은 파일 내용을 바이트로 반환합니다.
            content = gdown.download(url=download_url, quiet=True, fuzzy=True, output='-')
            if content is None:
                st.error("❌ 파일 다운로드에 실패했습니다. 파일 ID나 권한을 확인하세요.")
                return None
            
            pt_data = torch.load(BytesIO(content), map_location="cpu")
            st.success("✅ .pt 파일 로딩 성공!")
            return pt_data
    except Exception as e:
        st.error(f"❌ 파일 처리 중 오류 발생: {e}")
        return None

# --- UI 및 메인 로직 ---
st.title("📁 GDrive 기반 .pt 메타데이터 시각화")

uploaded_json = st.file_uploader("📥 메타데이터 JSON 파일 업로드 (final_metadata_with_gdrive_ids.json)", type="json")

if uploaded_json:
    df = load_json_data(uploaded_json)
    
    if df is not None:
        st.success(f"✅ {len(df)}개 메타데이터 로딩 완료")

        # --- 사이드바 필터 ---
        st.sidebar.header("🔍 필터 조건")
        table_ids = sorted(df['table_id'].dropna().unique().astype(int))
        labels = sorted(df['label'].dropna().unique().astype(int))
        
        selected_table = st.sidebar.selectbox("Table ID 선택", ["None"] + list(table_ids))
        selected_label = st.sidebar.selectbox("Label 선택", ["None"] + list(labels))

        # --- 데이터 필터링 ---
        filtered_df = df.copy()
        if selected_table != "None":
            filtered_df = filtered_df[filtered_df['table_id'] == selected_table]
        if selected_label != "None":
            filtered_df = filtered_df[filtered_df['label'] == selected_label]

        st.subheader("📂 필터링 결과")
        st.write(f"🔍 조건에 맞는 파일: {len(filtered_df)}개")
        
        # --- AgGrid 테이블 ---
        # 이 부분을 복사하여 기존 코드의 동일한 로직을 대체하세요.

        # --- AgGrid 테이블 ---
        if not filtered_df.empty:
            gb = GridOptionsBuilder.from_dataframe(filtered_df)
            gb.configure_selection('single', use_checkbox=False)
            grid_options = gb.build()
            
            grid_response = AgGrid(
                filtered_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                width='100%',
                allow_unsafe_jscode=True,
                key='aggrid_table'
            )

            # =================================================================
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 여기가 가장 중요한 부분입니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # -----------------------------------------------------------------
            # AgGrid로부터 '선택된 행' 정보를 '리스트' 형태로 받아옵니다.
            # grid_response.get('data')가 아닌 grid_response.get('selected_rows', [])를 사용해야 합니다.
            selected = grid_response.get('selected_rows', [])
            # -----------------------------------------------------------------
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # --- 선택된 파일 처리 ---
            # 이제 'selected'는 리스트이므로, if문으로 비어있는지 정상적으로 확인할 수 있습니다.
            if selected:
                # 리스트의 첫 번째 항목(선택된 행 정보가 담긴 딕셔너리)을 추출합니다.
                selected_row = selected[0]
                file_id = selected_row.get('gdrive_file_id')

                # 이전에 선택한 파일과 다를 경우에만 새로 다운로드
                if file_id != st.session_state.get('last_selected_id'):
                    st.session_state.pt_data = download_and_load_pt(file_id)
                    st.session_state.last_selected_id = file_id
                
                # --- 상세 정보 표시 ---
                # (이하 로직은 이전 답변과 동일하게 유지)
                if st.session_state.get('pt_data'):
                    st.subheader(f"📁 상세 정보: {selected_row.get('filename')}")
                    
                    pt_data = st.session_state.pt_data
                    
                    # (이미지 및 차트 표시 로직...)
                    # ...

            else:
                 # 선택이 해제되면 저장된 데이터 초기화
                st.session_state.pt_data = None
                st.session_state.last_selected_id = None


            # --- CSV 다운로드 버튼 ---
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 필터 결과 CSV 다운로드",
                data=csv_data,
                file_name="filtered_metadata.csv",
                mime="text/csv"
            )