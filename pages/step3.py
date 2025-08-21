import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time
import streamlit.components.v1 as components
from st_pages import Page, Section, show_pages, add_indentation
import shutil
import os


try:
    from pages.step2 import generate_course_section
except ImportError:
    from step2 import generate_course_section

add_indentation()

def clear_generation_cache():
    """clear the session state for generated course sections"""
    generation_keys = ['generated_course_1', 'generated_course_2', 'generated_course_3', 'generated_course_4']
    for key in generation_keys:
        if key in st.session_state:
            del st.session_state[key]

def main():
    st.header("📚 Generating Your Customized Course")

    if hasattr(st.session_state, 'paper_category'):
        category_display = {
            "EDUCATION_LEARNING_SCIENCE": "Education/Learning Science",
            "LINGUISTICS_THEORY": "Linguistics/Theory (Generating general language tutoring course)",
            "NON_EDUCATION": "Non-Education Field (Generating general tutoring strategies)"
        }
        st.info(f"📋 Paper Classification: {category_display.get(st.session_state.paper_category, 'Unknown')}")

    required_keys = ['key', 'text', 'template1', 'topic']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"❌ Missing required data: {missing_keys}")
        st.warning("Please go back and complete the setup process.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🏠 Go to Home"):
                switch_page("app")
        with col2:
            if st.button("📄 Go to Step 2"):
                switch_page("step2")
        return

    with st.expander("📋 Current Configuration", expanded=False):
        st.write(f"**Topic:** {st.session_state.get('topic', 'Not set')}")
        st.write(f"**Learning Objective:** {st.session_state.get('learning_objective', 'Default objectives will be used')}")
        st.write(f"**Template Type:** {'Math Content' if st.session_state.get('is_math_template', False) else 'General Tutoring'}")
        st.write(f"**Text Chunks:** {len(st.session_state.get('text', []))} chunks loaded")

    clear_generation_cache()
    if not st.session_state.get('generated_course_1'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Generating Course Introduction...")
            progress_bar.progress(10)

            template1 = st.session_state.template1
            if '{learning_objective}' in template1:
                formatted_template = template1.format(
                    learning_objective=st.session_state.get('learning_objective', 'Develop effective tutoring strategies')
                )
            else:
                formatted_template = template1

            st.session_state.generated_course_1 = generate_course_section(
                st.session_state.key, 
                st.session_state.text, 
                formatted_template
            )
            progress_bar.progress(30)
            
            # Scenario 1
            status_text.text("🔄 Generating Scenario 1...")
            st.session_state.generated_course_2 = generate_course_section(
                st.session_state.key,
                st.session_state.text,
                st.session_state.template2
            )
            progress_bar.progress(60)
            
            # Scenario 2
            status_text.text("🔄 Generating Scenario 2...")
            st.session_state.generated_course_3 = generate_course_section(
                st.session_state.key,
                st.session_state.text,
                st.session_state.template3
            )
            progress_bar.progress(85)
            
            # Research Insights
            status_text.text("🔄 Generating Research Insights...")
            st.session_state.generated_course_4 = generate_course_section(
                st.session_state.key,
                st.session_state.text,
                st.session_state.template4
            )
            progress_bar.progress(100)
            
            status_text.text("✅ Course generation completed!")
            time.sleep(1)

            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error generating course: {str(e)}")
            st.write("Please try again or check your settings.")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Retry Generation"):
                    clear_generation_cache()
                    st.rerun()
            with col2:
                if st.button("🏠 Go Back to Home"):
                    switch_page("app")
            return

    st.success("🎉 Course Generated Successfully!")

    with st.container():
        st.markdown("### 📖 Course Introduction")
        if st.session_state.get('generated_course_1'):
            st.write(st.session_state.generated_course_1)
        else:
            st.warning("Course introduction not available")

    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("⬅️ Previous", help="Go back to setup"):
            switch_page("step2")

    with col2:
        if st.button("🔄 Regenerate", help="Generate new content", key="regenerate_results"):
            clear_generation_cache()
            # 清理向量数据库
            for item in os.listdir("."):
                if item.startswith("chroma_db"):
                    try:
                        shutil.rmtree(item, ignore_errors=True)
                    except:
                        pass
            st.rerun()

    with col3:
        if st.button("➡️ Next", help="Go to Scenario 1"):
            switch_page("page 2: scenario 1")

    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("📖 Introduction", key="nav1"):
            switch_page("Page 1: Introduction")
            
    with nav_col2:
        if st.button("🎭 Scenario 1", key="nav2"):
            switch_page("page 2: scenario 1")
            
    with nav_col3:
        if st.button("🎪 Scenario 2", key="nav3"):
            switch_page("Page 3: Scenario 2")
            
    with nav_col4:
        if st.button("🔬 Research", key="nav4"):
            switch_page("Page 4: Research Says")

if __name__ == "__main__":
    main()