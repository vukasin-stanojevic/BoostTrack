import streamlit as st
from ui.roi_tab import roi_tab
from ui.track_speed_tab import track_tab

st.set_page_config(page_title="POC", layout="wide")
st.title("인천공항 POC")
st.write("info")

tab1, tab2, tab3 = st.tabs(["track_speed", "ROI", "도움"])

with tab1:
    track_tab()

with tab2:
    st.header("말")

with tab3:
    st.header("도움말")
    st.write("이 앱은 비디오에서 추출한 썸네일 위에 사각형 영역을 그리고, 해당 영역만 파란색으로 표시하는 기능을 제공합니다.")
    