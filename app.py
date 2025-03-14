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
    