import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, Section, show_pages, add_indentation

add_indentation()

st.write(st.session_state.generated_course_2)
# Navigation buttons
col1, col2 = st.columns([5,1])

with col1:
    if st.button("Previous"):
        switch_page("Page 1: Introduction")

with col2:
    if st.button("Next"):
        switch_page("Page 3: Scenario 2")
