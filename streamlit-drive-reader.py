import streamlit as st
import pandas as pd
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account

st.set_page_config(page_title="Google Drive PT 파일 ID 추출기", layout="centered")
st.title("🔐 Google Drive .pt 파일 ID 추출기")

st.markdown("""
이 앱은 Google Drive 공유 폴더 내 `.pt` 파일들의 **파일 이름과 File ID**를 추출하여 CSV로 제공합니다.

1. **서비스 계정 키(JSON)** 파일을 업로드하세요.
2. Google Drive에서 `.pt` 파일들이 들어있는 **폴더 ID**를 입력하세요.
3. 아래에서 **CSV 다운로드** 버튼으로 저장하세요.
""")

uploaded_json = st.file_uploader("1️⃣ 서비스 계정 키(JSON) 업로드", type="json")
folder_id = st.text_input("2️⃣ Google Drive 공유 폴더 ID 입력")

if uploaded_json and folder_id:
    try:
        service_account_info = json.load(uploaded_json)
        SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        st.info("📡 Google Drive에서 .pt 파일 목록 불러오는 중...")

        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'",
            fields="files(id, name)",
            pageSize=1000
        ).execute()

        files = results.get('files', [])
        pt_files = [(f['name'], f['id']) for f in files if f['name'].endswith('.pt')]

        if not pt_files:
            st.warning("⚠️ .pt 파일이 폴더에 존재하지 않습니다.")
        else:
            df = pd.DataFrame(pt_files, columns=["filename", "gdrive_file_id"])
            st.success(f"✅ 총 {len(df)}개의 .pt 파일을 찾았습니다!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV 다운로드", csv, file_name="pt_file_ids.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
