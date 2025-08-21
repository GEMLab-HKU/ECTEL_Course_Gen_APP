import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, Section, show_pages, add_indentation

add_indentation()

st.write(st.session_state.generated_course_3)
# Navigation buttons
col1, col2 = st.columns([5,1])

with col1:
    if st.button("Previous"): 
        # switch_page("Page 2: Scenario 2")
        switch_page("page 2: scenario 1")


with col2:
    if st.button("Next"):
        switch_page("Page 4: Research Says")
