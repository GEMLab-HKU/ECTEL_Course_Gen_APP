__import__('sqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('sqlite3')

import time
import uuid
import shutil
import os
import streamlit as st
import re
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import openai
from langchain_openai import ChatOpenAI
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, Section, show_pages, add_indentation

add_indentation()

show_pages([
    Page("app.py", "Create a new lesson", "🏠"),
    Section(name="Example Lessons", icon="🎈️"),
    Page("pages/example.py", name="example1", icon=":star:"),
    Page("pages/example2.py", name="example2", icon=":star:"),
    Section(name="Result", icon="🎈️"),
    Page("pages/step3.py", name="Page 1: Introduction"),
    Page("pages/step3_result2.py", name="Page 2: scenario 1"),
    Page("pages/step3_result3.py", name="Page 3: Scenario 2"),
    Page("pages/step3_result4.py", name="Page 4: Research Says")
])

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

template_1 = """
What tutoring strategy are the papers mainly talking about? Can you generate three components of a scenario-based tutor training course about how to use the tutoring strategy effectively in classrooms as discussed in the retrieved research paper?

I need your help to generate the course title, description, and learning objective, please follow the below template.

The course is about the topic the user input in online math tutoring session.

You should generate below 3 components, the title of the course, the course description, and the learning objectives, according to the requirements:

1.Course Title: Generate a title of this tutor training course using three words, the title should begin with a verb. It should be related to the specific topic of the research paper and the course objective.
Example titles are: Managing inequity, Managing effective praise

2.Description: A short description (50-60 words) about the purpose of this course and why it's important for the tutor. The structure could be similar to: Have you ever met a situation when you are in an online tutoring session, you find your students are [the background of the tutoring topic] and you want to change the situation? In this module, we will be introducing [strategy name] as a way of tutoring students in an online session more effectively.

3. Learning Objectives (15-20 words for each):
Requirement for generation: You should generate 2 learning objectives. The learning objectives should address the "understanding" and the "creating" level of Bloom's taxonomy. Understanding means Constructing meaning from oral, written, and graphic messages through interpreting, exemplifying, classifying, summarizing, inferring, comparing, and explaining.

Creating means putting elements together to form a coherent or functional whole; reorganizing elements into a new pattern or structure through generating, planning, or producing.

Objective 1: Describe the expected outcome of this course.
Objective 2: Outline what learners will achieve by the end of this module regarding the second objective.

One of the objectives should clearly state the most effective strategy about how to apply this tutoring strategy in online tutoring session that is advocated in the retrieved research paper.

Please give the reason why you design the objective like this.

You don't need to create the specific scenario right now.
"""

template_2 = """
Can you generate the scenario-based math tutor training course's first scenario about the below course title and learning objective based on the retrieved research paper? Please follow the below template to generate the first scenario for the training course based on the title and the learning objective.

Scenario-Based Online Tutor Training Course Development

Task: Generate the first scenario for an online tutor training course based on the provided course title and learning objective you generated in step 1, using the retrieved research paper. Follow the template below to structure the scenario.

Background information:

{ Output from Template 1:
Course Title: ()
Description: ()
Learning Objectives:()
}

You don't need to show the above information again in your output.

You need to generate below components:

Template:
Scenario Structure:
Scenario 1: Describe an initial training scenario involving a common situation related to the course title and learning objectives when a teacher is tutoring online.
Scenario Context: Create a scenario involving a challenge related to the topic of the paper when the teacher is tutoring a student named [Student Name A]. The scenario should focus on the student's response that relates to the course topic. Use approximately 50 words.
Questions:
1. Constructed-response Open-Ended Question (Motivation):
    * Question: Ask participants to propose their response or solution to the scenario, directly addressing the mathematical topic-related issue in the scenario.
    * Purpose: Initial reaction, free expression.
    * Reason: Encourages creative thinking and reflection.
2. Selected-response Question (Assessment of Understanding):
    * Question: Present four possible tutor responses in conversation style about how to teach math effectively, and all responses should reflect the usage of the research recommendations with varied appropriateness, that could be applied in the scenario. Ask tutors to choose the most effective option which is most related to the research recommendation of the papers.
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
    * Question: Present four possible tutor responses about how to teach this math topic effectively (statements, not conversations) that align with the responses in the previous questions, revealing the research-recommended strategy for the formative training scenario. Ask participants to select the principle that best supports their chosen response.
    * Requirements:
        * Provide statements reflecting various educational, ethical, or theoretical underpinnings related to the scenario.
        * Highlight the correct answer.
    * Options:
        * A. [Option A]
        * B. [Option B]
        * C. [Option C]
        * D. [Option D]
    * Mark the correct answer and explain the reason for its selection and why the others are incorrect.
"""

template_3 = """
Task: Generate the second scenario and the conclusion part for the online tutor training course based on the provided course title and learning objective you already generated, using the retrieved research paper.
Structure the scenario using the template below. It should be the same difficulty to answer as Scenario 1's questions.
The length will also be the same. Also, please generate the conclusion of this course based on the template given.

Background information (you don't need to generate this any more in your result):
{
{ [Your generated Output from Template 1]:
Course Title: ()
Description:()
Learning Objectives:()

[Your generated Output from Template 2]
Scenario 1 is:
{Scenario 1's description}
}

You don't need to show the above information again in your output.
You need to generate below components:

Template:
Scenario Structure:
Scenario 2: Describe a transfer training scenario involving a common situation related to the course topic the user input when he/she is tutoring math online. This scenario should involve a new student [use a different student name than in Scenario 1] and is designed for tutors who have completed the initial scenario, but still focuses on the same topic.
Scenario Context: Create a scenario involving a challenge related to the topic of the paper when the teacher is tutoring a student named B [a different student name, compared to what you have in scenario 1]. The scenario should focus on the student's response that relates to the course topic. Use approximately 50 words.

Questions:
1. Constructed-response Open-Ended Question (Motivation):
    * Question: Ask participants to propose their response or solution to the scenario, directly addressing the topic-related issue in the scenario.
    * Purpose: Initial reaction, free expression.
    * Reason: Encourages creative thinking and reflection.
2. Selected-response Question (Assessment of Understanding):
    * Question: Present four possible tutor responses in conversation style about how to teach math effectively, and all responses should reflect the usage of the research recommendations with varied appropriateness, that could be applied in the scenario. Ask tutors to choose the most effective option which is most related to the research recommendation of the papers.
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
    * Question: Present four possible tutor responses about how to teach this math topic effectively (statements, not conversations) that align with the responses in the previous questions, revealing the research-recommended strategy for the formative training scenario. Ask participants to select the principle that best supports their chosen response.
    * Requirements:
        * Provide statements reflecting various educational, ethical, or theoretical underpinnings related to the scenario.
        * Highlight the correct answer.
    * Options:
        * A. [Option A]
        * B. [Option B]
        * C. [Option C]
        * D. [Option D]
    * Mark the correct answer and explain the reason for its selection and why the others are incorrect.

Conclusion Summary
Write a brief closing (40-50 words) summarizing the key takeaways from the course. Reinforce the importance of the tutoring strategy and its impact on student learning outcomes.
"""

template_4 = """
Can you generate the scenario-based course's research insights part based on the retrieved research paper and your generated scenario based courses and the below template? 
Please follow the template I give you.
Please don't say 'I don't know'.
If there is not enough information, make reasonable assumptions based on general research on online tutoring strategies. 
Provide a detailed response following the structure below.

You are an AI research assistant. Your task is to analyze the provided research papers and extract information to fulfill the user's request. You MUST base your response solely and exclusively on the information contained in the retrieved research papers. You are NOT allowed to assume, infer, or generate any information that is not directly supported by the text.

If the required information is not present in the papers, you must state this clearly. 
In that case, you may provide a short fallback explanation based on established educational principles, 
but you must explicitly label it as "Not from the provided papers."

You should generate the following content:

Research Insights:

Summarize key research findings that support the learning objectives.
You should have at least 3 paragraphs to talk about these research findings, and add in-text citations.
Discuss practical applications of these insights.
An example could be as below, and you can use the same structure:

"Research says…
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

If the retrieved papers do not provide enough detail, create a default but relevant training course 
drawing from general tutoring strategies (e.g., scaffolding, questioning, feedback). 

Still follow the required structure.
"""

template_1_2 = """
Scenario-Based Online Tutor Training Course Development

Task: Generate the first scenario for an online tutor training course based on the provided course title and learning objective, using the retrieved research paper. Follow the template below to structure the scenario.

When explaining the answers, explicitly state why the correct option is aligned with the research 
and why each incorrect option is less effective or irrelevant.

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
Task: Generate the second scenario for an online tutor training course based on the provided course title and learning objective, using the retrieved research paper. 
Follow the template below to structure the scenario. It should be the same difficulty to answer as Scenario 1. 
The length will also be the same.


When explaining the answers, explicitly state why the correct option is aligned with the research 
and why each incorrect option is less effective or irrelevant.

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
You are an AI research assistant. Your task is to analyze the provided research papers and extract information to fulfill the user's request. 

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

If fewer than three distinct findings are available in the retrieved papers, 
supplement with well-established educational theories (clearly marked as not from the papers).  

At least one strategy in the Strategy Table must be directly drawn from the retrieved paper content. 
If citation details are incomplete, indicate this with "(Incomplete citation)".
"""

classification_prompt = """
Please analyze the provided text from a research paper and classify its primary field of study. 

Your response MUST be ONLY one of the following three labels:

1. "EDUCATION_LEARNING_SCIENCE" - If the paper is primarily about teaching methods, learning theories, pedagogical strategies, educational technology, teacher training, or student outcomes in any subject.

2. "LINGUISTICS_THEORY" - If the paper is primarily about the structure, history, or theory of a language itself (e.g., syntax, phonetics, semantics, sociolinguistics) WITHOUT a primary focus on pedagogy.

3. "NON_EDUCATION" - If the paper is primarily about any other field, such as medicine, physics, pure mathematics, chemistry, biology, economics, etc., and is not fundamentally about teaching or learning.

Do not provide explanations. Only output the single most relevant label.
Paper Text: {paper_text_extract}
"""


# # Add this function to classify PDF content
# def classify_pdf_content(text_chunks, openai_api_key):
#     """
#     Classify PDF content to determine if it's math-related or general education content
#     """
#     try:
#         # Create a simple classification prompt
#         classification_prompt = """
#         Analyze the provided text from research papers and determine if the primary content is:
#         1. Mathematics education or math tutoring specific content
#         2. General education/tutoring content (not math-specific)
#         3. Other content (not education-related)
#
#         Respond with only one of these three labels: "MATH", "GENERAL_EDUCATION", or "OTHER".
#         """
#
#         # Use a portion of the text for classification
#         sample_text = ""
#         for chunk in text_chunks[:5]:  # Use first 5 chunks for classification
#             sample_text += chunk.page_content[:500] + "\n\n"  # Take first 500 chars from each chunk
#
#         if len(sample_text) < 100:  # Not enough text to classify
#             return "GENERAL_EDUCATION"  # Default to general education
#
#         # Create embeddings and vector store for classification
#         embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         vector_db = Chroma.from_documents(
#             documents=[Document(page_content=sample_text)],
#             embedding=embeddings
#         )
#
#         # Create retriever
#         retriever = vector_db.as_retriever()
#
#         # Create classification chain
#         classification_template = """
#         You are an AI assistant that classifies educational research papers.
#         Based on the content below, classify whether the paper is primarily about:
#         - Mathematics education (respond with "MATH")
#         - General education/tutoring strategies (respond with "GENERAL_EDUCATION")
#         - Other content not related to education (respond with "OTHER")
#
#         Content: {context}
#
#         Your response should be only one of these three labels: "MATH", "GENERAL_EDUCATION", or "OTHER".
#         """
#
#         PROMPT = PromptTemplate(
#             template=classification_template,
#             input_variables=["context"]
#         )
#
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=ChatOpenAI(
#                 openai_api_key=openai_api_key,
#                 model_name="gpt-3.5-turbo",
#                 temperature=0,
#             ),
#             chain_type="stuff",
#             chain_type_kwargs={"prompt": PROMPT},
#             retriever=retriever,
#         )
#
#         result = qa_chain.invoke({"query": classification_prompt})
#         classification = result["result"].strip().upper()
#
#         # Clean up
#         vector_db.delete_collection()
#
#         # Validate classification
#         if classification in ["MATH", "GENERAL_EDUCATION", "OTHER"]:
#             return classification
#         else:
#             return "GENERAL_EDUCATION"  # Default to general education
#
#     except Exception as e:
#         st.error(f"Error classifying content: {str(e)}")
#         return "GENERAL_EDUCATION"  # Default to general education on error

def read_pdfs(pdf_files):

    docs = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        docs.append(Document(page_content=text))
    return docs

def split_text(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=100, separators=["\n\n", "\n\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents=docs)

def clear_all_cache():

    cache_keys = [
        # Keys from your original function
        'generated_course_1', 'generated_course_2',
        'generated_course_3', 'generated_course_4',
        'text', 'key', 'template1', 'template2',
        'template3', 'template4', 'is_math_template',

        # --- Add these keys to fully reset the state ---
        'topic', 'learning_objective'
    ]

    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]

    st.cache_resource.clear()

    for item in os.listdir("."):
        if item.startswith("chroma_db"):
            try:
                shutil.rmtree(item, ignore_errors=True)
            except Exception as e:
                st.warning(f"Could not clear cache directory: {e}")

# In your main() function in app.py, you have a button for this:
if st.button("🔄 Clear Cache & Start Fresh", key="clear_cache_home"):
    clear_all_cache()
    st.success("Cache cleared! Please refresh the page to apply changes.")
    st.rerun()
    """clear all cache and reset the state of the app"""
    cache_keys = [
        'generated_course_1', 'generated_course_2',
        'generated_course_3', 'generated_course_4',
        'text', 'key', 'template1', 'template2',
        'template3', 'template4', 'is_math_template'
    ]

    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]

    st.cache_resource.clear()

    persist_directory = "chroma_db"
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            st.warning(f"Could not clear cache directory: {e}")

def generate_course_section_no_cache(openai_api_key, text_chunks, template):
    """generate course section without using cache"""

    persist_directory = f"chroma_db_{uuid.uuid4().hex[:8]}_{int(time.time())}"

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_db = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        custom_retriever = vector_db.as_retriever()
        custom_retriever.search_type = "mmr"
        custom_retriever.search_kwargs = {"fetchK": 10, "lambda": 0.25}

        prompt_template_text = """
        You are an expert course generator tasked with creating a comprehensive tutor training course for online tutors. Use the retrieved information to answer the question and follow the template the user gives you, but you should not refer to the template context. Please generate a course based on the research paper. The users of the course are novice tutors who are experts in the subjects they are teaching but unfamiliar with the best method of teaching what they know to the students. The course should have a course title and some practical examples (each scenario should have 4 questions) and some practical research recommendation examples for the tutor to show how they could perform in their sessions.

        For the contents you generated, please make sure the Flesch-Kincaid Score is below 10.

        If you don't know the answer, please say "Sorry, I don't have enough context on this, please give me more instructions and I will try my best to help you!"
        {context}
        Question: {question}
        """

        PROMPT = PromptTemplate(template=prompt_template_text, input_variables=["context", "question"])

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
        st.error(f"Error in generate_course_section_no_cache: {str(e)}")
        return f"Error generating content: {str(e)}"

    finally:

        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
            except Exception as e:
                st.warning(f"Could not clean up temporary directory: {e}")

def main():
    st.header("Create new lessons for online math tutors! ")
    st.text("Input your customized course information")


    if 'topic' not in st.session_state:
        st.session_state.topic = None

    if 'learning_objective' not in st.session_state:
        st.session_state.learning_objective = ""

    topics = ["Social Emotional Learning", "Mastery of Content", "Advocacy",
              "Building Relationships", "Utilizing Technology Tools", "Other"]

    current_index = None
    if 'topic' in st.session_state and st.session_state.topic:
        if st.session_state.topic in topics:
            current_index = topics.index(st.session_state.topic)
        else:
            current_index = topics.index("Other")

    selected_topic = st.selectbox(
        "Select a topic",
        topics,
        index=current_index,
        key="topic_selector"
    )

    st.session_state.topic = selected_topic

    if st.session_state.topic == "Other":
        custom_topic = st.text_input(
            "Enter your custom topic:",
            value="",
            key="custom_topic_input"
        )
        if custom_topic.strip():
            st.session_state.topic = custom_topic.strip()

    if st.session_state.topic == "Mastery of Content":
        st.session_state.is_math_template = True
        st.info("📚 Math content templates will be used.")
    elif st.session_state.topic and st.session_state.topic != "Other":
        st.session_state.is_math_template = False
        st.info("🎯 General tutoring templates will be used.")

    example_text = """What's the lesson's learning objective? (optional)                                              
    (Example: 1. Identify features of tutors encouraging students' independence when engaging in tutoring  
    2. Explain the importance of encouraging students' independence when working with students  
    3. Apply strategies to encourage students' independence)"""

    st.session_state.learning_objective = st.text_input(
        example_text,
        value=st.session_state.learning_objective,
        key="learning_objective_input"
    )

    openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
    uploaded_files = st.file_uploader(
        "Upload PDF files (please enter less than five files)",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    with st.expander("📋 Current Settings"):
        st.write(f"**Topic:** {st.session_state.topic}")
        st.write(f"**Learning Objective:** {st.session_state.learning_objective}")
        st.write(f"**Template Type:** {'Math Content' if st.session_state.get('is_math_template', False) else 'General Tutoring'}")
        st.write(f"**Files Uploaded:** {len(uploaded_files) if uploaded_files else 0}")

    col1, col2 = st.columns([3.1, 1.5])

    with col1:
        if st.button("Visit pre-generated example lesson"):
            switch_page("example2")

    with col2:
        # This button now simply validates input and moves to the next step.
        if st.button("Next: Upload Files"):
            if not st.session_state.topic or st.session_state.topic == "Other":
                st.error("Please select or enter a valid topic")
            elif not openai_api_key:
                st.error("Please enter your OpenAI API Key")
            elif not uploaded_files:
                st.error("Please upload at least one PDF file")
            else:
                # All good, now process the files and save critical info to session_state
                # before moving to the next page.
                try:
                    # Save the key now
                    st.session_state.key = openai_api_key

                    # Process and save the text chunks
                    docs = read_pdfs(uploaded_files)
                    text_chunks = split_text(docs)
                    st.session_state.text = text_chunks

                    # Set the templates based on the topic
                    if st.session_state.get('is_math_template', False):
                        st.session_state.template1 = template_1
                        st.session_state.template2 = template_2
                        st.session_state.template3 = template_3
                        st.session_state.template4 = template_4
                    else:
                        st.session_state.template1 = template_1_1
                        st.session_state.template2 = template_1_2
                        st.session_state.template3 = template_1_3
                        st.session_state.template4 = template_1_4

                    st.success("Settings saved! Proceeding to generation...")
                    time.sleep(1) # Give user a moment to see the success message
                    switch_page("Page 1: Introduction") # Go directly to the generation page

                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key")
            return

        if not uploaded_files:
            st.error("Please upload at least one PDF file")
            return

        if not st.session_state.topic or st.session_state.topic == "Other":
            st.error("Please select or enter a valid topic")
            return

        clear_all_cache()

        try:
            docs = read_pdfs(uploaded_files)
            text_chunks = split_text(docs)

            # session state
            st.session_state.key = openai_api_key
            st.session_state.text = text_chunks

            if st.session_state.get('is_math_template', False):
                # math
                st.session_state.template1 = template_1
                st.session_state.template2 = template_2
                st.session_state.template3 = template_3
                st.session_state.template4 = template_4
            else:
                # general
                st.session_state.template1 = template_1_1
                st.session_state.template2 = template_1_2
                st.session_state.template3 = template_1_3
                st.session_state.template4 = template_1_4

            st.success(f"✅ Settings saved! Generating course for topic: {st.session_state.topic}")
            switch_page("Page 1: Introduction")

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            return

if __name__ == "__main__":
    main()
