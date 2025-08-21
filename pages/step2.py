import streamlit as st
import shutil
import os
import time
import uuid

# LangChain 相关
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings  # 已切换到 langchain-community
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import ChatOpenAI  # 已切换到 langchain-openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF 处理
from PyPDF2 import PdfReader

# Streamlit 页面辅助
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, Section, show_pages, add_indentation


add_indentation()

# Define your prompt template
prompt_template = """
You are an expert course generator tasked with creating a comprehensive tutor training course for online tutors. Use the retrieved information to answer the question and follow the template the user gives you, but you should not refer to the template context. Please generate a course based on the research paper. The users of the course are novice tutors who are experts in the subjects they are teaching but unfamiliar with the best method of teaching what they know to the students. The course should have a course title and some practical examples (each scenario should have 4 questions) and some practical research recommendation examples for the tutor to show how they could perform in their classrooms.

For the contents you generated, please make sure the Flesch-Kincaid Score is below 10.

If you don't know the answer, just say "I don't know".
{context}
Question: {question}
"""

# Math Templates (for Mastery of Content)
template_1 = """
Do you know what math topic the paper is talking about? Can you generate a scenario-based tutor training course about how to teach the mathematic topic effectively as discussed in the retrieved research paper?

I need your help to generate the course title, description and learning objectives, please follow the below template.

Course Title: Generate a title of this math course using three words, the title should begin with a verb. It should be related to the specific math topic of the research paper and the course objective.
Example titles are: Using polite language, Managing inequity, Managing effective praise

Description: A short description about the purpose of this course and why. The structure could be similar to: Have you ever met a situation where you want to teach [the math topic] but you find yourself unable to explain the concept clearly to the students? In this module, we will be introducing [strategy name] as a way of tutoring students about [the math topic].

Learning Objectives:
You should generate based on the given learning objective here {learning_objective}
Requirement for generation: You should generate 2 learning objectives. The learning objectives should address the "understanding" and the "creating" level of Bloom's taxonomy.
Creating means use information to create something new, understanding means grasp meaning of instructional materials. You should generate two learning objectives.

Objective 1: Describe the expected outcome of this course.
Objective 2: Outline what learners will achieve by the end of this module regarding the second objective.

One of the objectives should clearly state the most effective strategy about teaching this math topic that is advocated in the retrieved research paper.

You don't need to generate the specific scenario at this time.
"""

template_2 = """
Can you generate the scenario-based math tutor training course's first scenario about the below course title and learning objective based on the retrieved research paper? Please follow the below template to generate the first scenario for the training course based on the title and the learning objective.

Please generate the math equation directly, rather than giving word in latex format like 'fraction'.

Background information:

{Course title + Description + Learning Objective generated}

You don't need to show the above information again in your output.

You need to generate below components:

Scenario Description:

Scenario 1: [The Initial Training Scenario describing a common challenge or issue a student is facing when a teacher is tutoring this math topic]

Context: Please generate a scenario involving a challenge related to the math topic of the paper for a student named [Student Name]. 
The scenario should be about the student encountering a problem or challenge related to [the specific math topic or situation], and their response to the challenge. 
Focus on what part of learning the math topic makes the student feel frustrated.

Scenario Context (3-4 sentences): [Generated scenario context]

Questions: Design Four New Questions about this Scenario

1. Constructed-response Open-Ended Question (Motivation):
Question: Ask participants to propose their response or solution to the scenario, directly addressing the mathematical topic related issue in the scenario.
Purpose: Initial reaction, free expression.
Reason: Encourages creative thinking and reflection.

2. Selected-response Question (Assessment of Understanding):
Question: Present 4 possible tutor responses about how to teach this math topic effectively with varied appropriateness that could be applied in the scenario. Ask tutors to choose the option they believe is most effective. Mark the correct response and give the reason why the others are not correct.
4 Options: The options should all be practical real-life conversations for the tutor to teach the math topic to the student in an online math tutoring session.
Requirements:
* Make one option the correct one.
* Among the three distractors, one should be obviously wrong/unrelated to the situation.
* The other two should be close distractions that would seem appropriate in other situations but not aligned with the recommendation of the paper.
* Ensure all four options are similar in length.
* Consider using some real-life math manipulatives in your examples (e.g., pizza, cup, cake) related to the strategy.
Options:
* A. [Option A]
* B. [Option B]
* C. [Option C]
* D. [Option D]
Mark the correct one and explain why the others are incorrect.

3. Constructed-response Open-Ended Question (Justification):
Question: Ask participants to explain why they chose the specific option in the previous question, detailing the reasoning behind their selection.
Purpose: Encourage deep reasoning and reflection to reinforce the tutor's understanding and justification.
Reason: This question aims to delve into the thought process of participants, ensuring they can justify their choices and demonstrate a solid grasp of effective teaching strategies related to the scenario.

4. Matching Questions
Please design a matching question that aligns with expert-recommended tutoring strategies. The question should include a table with two columns. The first column contains a realistic conversation between a tutor and a student about a math topic. The second column contains a set of multiple-choice questions, each highlighting a part of the tutor's conversation. The student needs to match these highlighted parts with the appropriate tutoring strategy. Use the following structure:
* Provide a realistic conversation between a tutor and a student, including multiple steps to help resolve the student's challenge.
* Highlight three parts of the conversation where the tutor is using specific strategies to teach the math topic.
* Create a menu with four options representing real-world effective strategies to teach the topic. And the strategy mentioned here in the options should be consistent with the research insights table I need your help to generate later. 
* For each highlighted part of the conversation, repeat the menu of options and ask the student to select the strategy that most closely matches the conversation part.
* Indicate the correct answer and provide reasons for the incorrect options.

Example Table Template:

| Conversation   | Options |
| -------- | ------- |
| A conversation happening with tutor and the student, including multiple steps to help resolve the challenge of the student. Highlight 3 parts of the conversation where the tutor is using some strategy to tutor the student about the math topic. | Menu with: 4 options about the real-world effective strategy to teach this topic related to the 3 highlighted conversation parts. And the strategy mentioned here in the options should be consistent with the research insights table I need your help to generate later. (Correct Answer: the strategy that most matches this part of the conversation), (Incorrect Ones: please state the reason). For other parts of the conversation, repeat the same menu with the same options. Repeat the MCQ menu 3 times. |
"""

template_3 = """
Can you generate the scenario-based math tutor training course's second scenario about the below course title and learning objective based on the retrieved research paper? Please follow the below template to generate the second scenario for the training course based on the title and the learning objective.
The situation of another student [use a new student name] in Scenario 2 will be different from Scenario 1 but analogous to it. That is, it should have different surface features but should be about math tutoring and address the same learning objectives.
It should be the same difficulty to answer as Scenario 1. The length will also be the same.

Background information:
{Course title + Description + Learning Objective generated + Scenario 1 generated}

You don't need to show the above information again in your output. You need to generate a completely different scenario, the structure will be as below.

You need to generate below components:

Scenario 2: [The Transfer Training Scenario describing a common challenge or issue a student is facing when a teacher is tutoring this math topic]

Context: Please generate a scenario involving a challenge related to the math topic of the paper for a student named [New Student Name]. 
The scenario should be about the student encountering a problem or challenge related to [the specific math topic or situation], and their response to the challenge. 
Focus on what part of learning the math topic makes the student feel frustrated.

Scenario Context (3-4 sentences): [Generated scenario context]

Questions: Design Four New Questions about this Scenario

1. Constructed-response Open-Ended Question (Motivation):
Question: Ask participants to propose their response or solution to the scenario, directly addressing the mathematical topic related issue in the scenario.
Purpose: Initial reaction, free expression.
Reason: Encourages creative thinking and reflection.

2. Selected-response Question (Assessment of Understanding):
Question: Present 4 possible tutor responses about how to teach this math topic effectively with varied appropriateness that could be applied in the scenario. Ask tutors to choose the option they believe is most effective. Mark the correct response and give the reason why the others are not correct.
4 Options: The options should all be practical real-life conversations for the tutor to teach the math topic to the student in an online math tutoring session.
Requirements:
* Make one option the correct one.
* Among the three distractors, one should be obviously wrong/unrelated to the situation.
* The other two should be close distractions that would seem appropriate in other situations but not aligned with the recommendation of the paper.
* Ensure all four options are similar in length.
* Consider using some real-life math manipulatives in your examples (e.g., pizza, cup, cake) related to the strategy.
Options:
* A. [Option A]
* B. [Option B]
* C. [Option C]
* D. [Option D]
Mark the correct one and explain why the others are incorrect.

3. Constructed-response Open-Ended Question (Justification):
Question: Ask participants to explain why they chose the specific option in the previous question, detailing the reasoning behind their selection.
Purpose: Encourage deep reasoning and reflection to reinforce the tutor's understanding and justification.
Reason: This question aims to delve into the thought process of participants, ensuring they can justify their choices and demonstrate a solid grasp of effective teaching strategies related to the scenario.

4. Matching Questions
Please design a matching question that aligns with expert-recommended tutoring strategies. The question should include a table with two columns. The first column contains a realistic conversation between a tutor and a student about a math topic. The second column contains a set of multiple-choice questions, each highlighting a part of the tutor's conversation. The student needs to match these highlighted parts with the appropriate tutoring strategy. Use the following structure:
* Provide a realistic conversation between a tutor and a student, including multiple steps to help resolve the student's challenge.
* Highlight three parts of the conversation where the tutor is using specific strategies to teach the math topic.
* Create a menu with four options representing real-world effective strategies to teach the topic. And the strategy mentioned here in the options should be consistent with the research insights table I need your help to generate later. 
* For each highlighted part of the conversation, repeat the menu of options and ask the student to select the strategy that most closely matches the conversation part.
* Indicate the correct answer and provide reasons for the incorrect options.

Example Table Template:

| Conversation   | Options |
| -------- | ------- |
| A conversation happening with tutor and the student, including multiple steps to help resolve the challenge of the student. Highlight 3 parts of the conversation where the tutor is using some strategy to tutor the student about the math topic. | Menu with: 4 options about the real-world effective strategy to teach this topic related to the 3 highlighted conversation parts. And the strategy mentioned here in the options should be consistent with the research insights table I need your help to generate later. (Correct Answer: the strategy that most matches this part of the conversation), (Incorrect Ones: please state the reason). For other parts of the conversation, repeat the same menu with the same options. Repeat the MCQ menu 3 times. |
"""

template_4 = """
Can you generate the scenario-based course's research insights part based on the retrieved research paper and the below information? Please follow the template I give you.

Information:

You should generate the below content:

Research Insights:

Summarize key research findings that support the learning objectives.
You should have at least 3 paragraphs to talk about these research findings, and add in-text citations.
Discuss practical applications of these insights.
An example could be as below and you can use the same structure:

"Research says…
{Research paper context}
{Summary}
"

Strategy Table:
Generate a table with three rows and four columns based on the topic of [Learning Objective of the course] according to the research recommendations. Each row should include the following:

Strategy: [Specify the strategy about the topic].
Description: [Provide a brief description of the strategy and its effectiveness in communication.]
Good Example: [Give an example demonstrating how the strategy can be applied in a tutoring scenario, including the tone of a tutor. You should also list the reason why it is correct or not correct]
Bad Example: [Give an example demonstrating how the strategy can be applied in a tutoring scenario, including the tone of a tutor. You should provide one incorrect example here, to be opposite to a good example. You should also list the reason why it is correct or not correct]

References:
Cite all scholarly references and sources used in developing this course. You should list the source of the research papers you use here.
Do not list sources you didn't use.
"""

# General Templates (for other topics)
template_1_1 = """
Do you know what tutoring strategy the paper is talking about? Can you generate a scenario-based tutor training course about how to use the tutoring strategy effectively in classrooms as discussed in the retrieved research paper?

I need your help to generate the course title, description and learning objective, please follow the below template.

Course Title: Generate a title of this tutor training course using three words, the title should begin with a verb. It should be related to the specific tutoring strategy of the research paper and the course objective.
Example titles are: Using polite language, Managing inequity, Managing effective praise

Description: A short description (50-60 words) about the purpose of this course and why it's important for the tutor. The structure could be similar to: Have you ever met a situation when you are in an online tutoring session, you find your students are [the background of the tutoring topic] and you want to change the situation? In this module, we will be introducing [strategy name] as a way of tutoring students in an online session more effectively.

Learning Objectives (15-20 words for each):
Requirement for generation: You should generate 2 learning objectives. The learning objectives should address the "understanding" and the "creating" level of Bloom's taxonomy.
Creating means use information to create something new, understanding means grasp meaning of instructional materials. You should generate two learning objectives.

Objective 1: Describe the expected outcome of this course.
Objective 2: Outline what learners will achieve by the end of this module regarding the second objective.

One of the objectives should clearly state the most effective strategy about how to apply this tutoring strategy in online tutoring session that is advocated in the retrieved research paper.
"""

template_1_2 = """
Generate the first scenario for an online tutor training course based on the course topic, using the retrieved research paper. Follow the template below to structure the scenario.

You need to generate below components:

Template:
Scenario Structure:
Scenario 1: Describe an initial training scenario involving a common situation related to the course topic when a teacher is tutoring online.
Scenario Context: Create a scenario involving a challenge related to the topic of the paper when the teacher is tutoring a student named [Student Name]. The scenario should focus on the student's response that relates to the course topic. Use approximately 50 words.

Questions:
1. Constructed-response Open-Ended Question (Motivation):
    * Question: Ask participants to propose their response or solution to the scenario, directly addressing the tutoring-related issue in the scenario.
    * Purpose: Initial reaction, free expression.
    * Reason: Encourages creative thinking and reflection.

2. Selected-response Question (Assessment of Understanding):
    * Question: Present four possible tutor responses in conversation style about how to handle the situation effectively, with varied appropriateness, that could be applied in the scenario. Ask tutors to choose the most effective option.
    * Requirements:
        * Make one option correct, aligning with the paper's suggestion.
        * One option should be obviously wrong/unrelated.
        * Two options should be close distractions but not aligned with the paper's recommendation.
        * Ensure all options are similar in length (20-30 words each).
    * Options:
        * A. [Option A]
        * B. [Option B]
        * C. [Option C]
        * D. [Option D]
    * Mark the correct one and explain why the others are incorrect.

3. Constructed-response Open-Ended Question (Justification):
    * Question: Ask participants to explain why they chose the specific option in the previous question, detailing the reasoning behind their selection.
    * Purpose: Encourage deep reasoning and reflection to reinforce the tutor's understanding and justification.
    * Reason: Ensures participants can justify their choices and demonstrate a solid grasp of effective teaching strategies related to the scenario.

4. Selected-response Question (Assessment of Understanding):
    * Question: Present four possible tutor responses about how to handle this situation effectively (statements, not conversations) that align with the responses in the previous questions, revealing the research-recommended strategy for the formative training scenario. Ask participants to select the principle that best supports their chosen response.
    * Requirements:
        * Provide statements reflecting various educational, ethical, or theoretical underpinnings related to the scenario.
        * Highlight the correct answer.
    * Options:
        * A. [Option A]
        * B. [Option B]
        * C. [Option C]
        * D. [Option D]
    * Mark the correct answer and explain the reason for its selection.
"""

template_1_3 = """
Generate the second scenario for an online tutor training course based on the course topic, using the retrieved research paper. 
Follow the template below to structure the scenario. It should be the same difficulty to answer as Scenario 1. 
The length will also be the same.

Template:
Scenario Structure:
Scenario 2: Describe a transfer training scenario involving a common situation related to the course topic when a teacher is tutoring online. This scenario should involve a new student [use a different student name than in Scenario 1] and is designed for tutors who have completed the initial scenario, but still focuses on the same topic.
Scenario Context: Create a scenario involving a challenge related to the topic of the paper when the teacher is tutoring a student named [Student Name]. The scenario should focus on the student's response that relates to the course topic. Use approximately 50 words.

Questions:
1. Constructed-response Open-Ended Question (Motivation):
    * Question: Ask participants to propose their response or solution to the scenario, directly addressing the tutoring-related issue in the scenario.
    * Purpose: Initial reaction, free expression.
    * Reason: Encourages creative thinking and reflection.

2. Selected-response Question (Assessment of Understanding):
    * Question: Present four possible tutor responses in conversation style about how to handle the situation effectively, with varied appropriateness, that could be applied in the scenario. Ask tutors to choose the most effective option.
    * Requirements:
        * Make one option correct, aligning with the paper's suggestion.
        * One option should be obviously wrong/unrelated.
        * Two options should be close distractions but not aligned with the paper's recommendation.
        * Ensure all options are similar in length (20-30 words each).
    * Options:
        * A. [Option A]
        * B. [Option B]
        * C. [Option C]
        * D. [Option D]
    * Mark the correct one and explain why the others are incorrect.

3. Constructed-response Open-Ended Question (Justification):
    * Question: Ask participants to explain why they chose the specific option in the previous question, detailing the reasoning behind their selection.
    * Purpose: Encourage deep reasoning and reflection to reinforce the tutor's understanding and justification.
    * Reason: Ensures participants can justify their choices and demonstrate a solid grasp of effective teaching strategies related to the scenario.

4. Selected-response Question (Assessment of Understanding):
    * Question: Present four possible tutor responses about how to handle this situation effectively (statements, not conversations) in the previous questions, revealing the research-recommended strategy for the formative training scenario. Ask participants to select the principle that best supports their chosen response.
    * Requirements:
        * Provide statements reflecting various educational, ethical, or theoretical underpinnings related to the scenario.
        * Highlight the correct answer.
    * Options:
        * A. [Option A]
        * B. [Option B]
        * C. [Option C]
        * D. [Option D]
    * Mark the correct answer and explain the reason for its selection.
"""

template_1_4 = """
Can you generate the scenario-based course's research insights part based on the retrieved research paper and the below information? Please follow the template I give you.

You should generate the below content:

Research Insights:

Summarize key research findings that support the learning objectives.
You should have at least 3 paragraphs to talk about these research findings, and add in-text citations.
Discuss practical applications of these insights.
An example could be as below and you can use the same structure:

"Research says…
{context}
{summary}
"

Strategy Table:
Generate a table with three rows and four columns based on the topic of [Learning Objective of the course] according to the research recommendations. Each row should include the following:

Strategy: [Specify the strategy about the topic].
Description: [Provide a brief description of the strategy and its effectiveness in communication.]
Good Example: [Give an example demonstrating how the strategy can be applied in a tutoring scenario, including the tone of a tutor. You should also list the reason why it is correct or not correct]
Bad Example: [Give an example demonstrating how the strategy can be applied in a tutoring scenario, including the tone of a tutor. You should provide one incorrect example here, to be opposite to a good example. You should also list the reason why it is correct or not correct]

References:
Cite all scholarly references and sources used in developing this course. You should list the source of the research papers you use here.
Do not list sources you didn't use.
"""

# Function to read PDFs and extract text
def read_pdfs(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        docs.append(Document(page_content=text))
    return docs

# Function to split text into chunks
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=100, separators=["\n\n", "\n\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents=docs)

def clear_session_cache():
    """clear session state"""
    cache_keys = [
        'generated_course_1', 'generated_course_2', 
        'generated_course_3', 'generated_course_4',
        'persist_directory'
    ]
    
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]

def generate_course_section(openai_api_key, text_chunks, template):
    """generate course section based on the provided text chunks and template"""

    unique_id = f"{uuid.uuid4().hex[:8]}_{int(time.time())}"
    persist_directory = f"chroma_db_{unique_id}"
    
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory, ignore_errors=True)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_db = Chroma.from_documents(
            documents=text_chunks, 
            embedding=embeddings, 
            persist_directory=persist_directory
        )

        custom_retriever = vector_db.as_retriever()
        custom_retriever.search_type = "mmr"
        custom_retriever.search_kwargs = {"fetchK": 10, "lambda": 0.25}
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qachain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name="gpt-4o",
                temperature=0.6,
                verbose=False,
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
            retriever=custom_retriever,
        )

        doc_prompt = qachain.invoke({"query": template})
        return doc_prompt["result"]
        
    except Exception as e:
        st.error(f"generate course section failed: {str(e)}")
        return f"generate course section failed: {str(e)}"
        
    finally:
        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory, ignore_errors=True)
            except Exception as e:
                st.warning(f"clear cache directory failed: {str(e)}")

# Streamlit interface
def main():
    st.text("What are you looking to learn today?")
    st.header("Input your customized course topic")

    current_step = st.progress(100)

    col1, col2 = st.columns([1,1])
    with col1:
        st.page_link("app.py", label="State your learning goal", icon="1️⃣")

    with col2:
        st.page_link("pages/step2.py", label="Upload your research papers", icon="2️⃣")

    if st.button("🔄 Clear All Cache", key="clear_cache_step2"):
        for item in os.listdir("."):
            if item.startswith("chroma_db"):
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except:
                    pass

        clear_session_cache()
        st.success("Cache cleared successfully!")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDF files (please enter less than five files)", type=["pdf"], accept_multiple_files=True)

    if 'topic' in st.session_state:
        st.info(f"📌 Current Topic: {st.session_state.topic}")
    if 'learning_objective' in st.session_state and st.session_state.learning_objective:
        st.info(f"🎯 Learning Objective: {st.session_state.learning_objective}")

    col1, col2 = st.columns([3.5,1.5])
    with col1:
        if st.button("Back"): 
            switch_page("app")

    with col2:
        generate_button = st.button("Generate Course Section")

    if generate_button:
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key")
            return
            
        if not uploaded_files:
            st.error("Please upload at least one PDF file")
            return
            
        if 'topic' not in st.session_state or not st.session_state.topic:
            st.error("Please go back and select a topic first")
            return

        clear_session_cache()

        for item in os.listdir("."):
            if item.startswith("chroma_db"):
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except:
                    pass

        try:
            docs = read_pdfs(uploaded_files)
            text_chunks = split_text(docs)

            # session_state
            st.session_state.key = openai_api_key
            st.session_state.text = text_chunks

            if st.session_state.get('topic') == "Mastery of Content":
                st.session_state.is_math_template = True
                st.session_state.template1 = template_1
                st.session_state.template2 = template_2
                st.session_state.template3 = template_3
                st.session_state.template4 = template_4
                st.success("📚 Using Math Content templates")
            else:
                st.session_state.is_math_template = False
                st.session_state.template1 = template_1_1
                st.session_state.template2 = template_1_2
                st.session_state.template3 = template_1_3
                st.session_state.template4 = template_1_4
                st.success("🎯 Using General Tutoring templates")

            if not st.session_state.get('learning_objective'):
                st.session_state.learning_objective = "Explain and create effective instructional strategies for student engagement and learning"

            st.success(f"✅ Ready to generate course for: {st.session_state.topic}")
            switch_page("step3")

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            return

if __name__ == "__main__":
    main()
