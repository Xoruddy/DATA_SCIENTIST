import tempfile
import os
import streamlit as st

from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

__import__('pysqlite3') 
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma 
os.environ["OPENAI_API_KEY"]=st.secrets['OPENAI_API_KEY']

# Streamlit 페이지 설정
st.set_page_config(page_title="벼락치기 도우미", page_icon="⏳")

def main():
    st.title("⏳ 대학생 벼락치기 도우미")

    if "uploaded_text" not in st.session_state:
        st.session_state.uploaded_text = None

    if "roadmap" not in st.session_state:
        st.session_state.roadmap = None

    if "quiz" not in st.session_state:
        st.session_state.quiz = None

    if "quiz_solutions" not in st.session_state:
        st.session_state.quiz_solutions = None

    if "evaluation_scores" not in st.session_state:
        st.session_state.evaluation_scores = None

    with st.sidebar:
        uploaded_files = st.file_uploader("📄 강의 자료 업로드", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API 키", type="password")
        exam_date = st.date_input("📅 시험 날짜를 선택하세요")
        process_button = st.button("🚀 벼락치기 시작하기")
        create_summary = st.checkbox("핵심 요약 생성", value=True)
        create_roadmap = st.checkbox("공부 로드맵 생성", value=True)
        create_quiz = st.checkbox("예상 문제 생성", value=True)

    if process_button:
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요!")
            return
        if not uploaded_files:
            st.warning("강의 자료를 업로드해주세요!")
            return
        if not exam_date:
            st.warning("시험 날짜를 선택해주세요!")
            return

        days_left = (exam_date - datetime.now().date()).days
        if days_left <= 0:
            st.warning("시험 날짜는 오늘보다 이후여야 합니다!")
            return

        st.session_state.uploaded_text = extract_text_from_files(uploaded_files)
        text_chunks = split_text_into_chunks(st.session_state.uploaded_text)
        vectorstore = create_vectorstore(text_chunks)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

        if create_summary:
            st.session_state.summary = summarize_text(text_chunks, llm)
        if create_roadmap:
            st.session_state.roadmap = create_study_roadmap(st.session_state.summary, llm, days_left)
        if create_quiz:
            st.session_state.quiz, st.session_state.quiz_solutions = generate_quiz_with_solutions(st.session_state.summary, llm)

    if st.session_state.uploaded_text:
        if create_summary:
            st.subheader("📌 핵심 요약")
            st.markdown(st.session_state.summary)

        if create_roadmap:
            st.subheader("📋 공부 로드맵")
            st.markdown(st.session_state.roadmap)
            visualize_roadmap(st.session_state.roadmap, days_left)

        if create_quiz:
            st.subheader("❓ 예상 문제")
            for i, question in enumerate(st.session_state.quiz):
                with st.expander(f"문제 {i+1}"):
                    st.markdown(question)
                    st.text_input(f"풀이 {i+1}", key=f"solution_{i}")

            if st.button("📝 제출 및 평가"):
                evaluate_answers(st.session_state.quiz_solutions)

            if st.session_state.evaluation_scores:
                st.subheader("📊 평가 성능 지표")
                st.write(f"정확도: {st.session_state.evaluation_scores['accuracy']*100:.2f}%")
                st.write(f"총 점수: {st.session_state.evaluation_scores['total_score']}")

# 추가된 함수들

def extract_text_from_files(files):
    doc_list = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
            tmp_file.write(file.read())
            temp_file_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_file_path)
        else:
            st.warning(f"지원하지 않는 파일 형식입니다: {file.name}")
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)

        os.remove(temp_file_path)
    return doc_list

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(text)

def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return FAISS.from_documents(text_chunks, embeddings)

def summarize_text(text_chunks, llm, max_summary_length=2000):
    def process_chunk(chunk):
        text = chunk.page_content
        messages = [
            SystemMessage(content="당신은 유능한 한국어 요약 도우미입니다."),
            HumanMessage(content=f"다음 텍스트를 한국어로 요약해주세요:\n\n{text}")
        ]
        response = llm(messages)
        return response.content

    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(process_chunk, text_chunks))

    combined_summary = "\n".join(summaries)
    return combined_summary[:max_summary_length] + "..." if len(combined_summary) > max_summary_length else combined_summary

def create_study_roadmap(summary, llm, days_left):
    if len(summary) > 2000:
        summary = summary[:2000] + "..."
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 유능한 공부 로드맵 작성 도우미입니다."),
        HumanMessage(content=f"다음 텍스트를 기반으로 {days_left}일 동안 한국 대학생들이 효과적으로 공부할 수 있는 계획을 작성해주세요:\n{summary}")
    ]
    response = llm(messages)
    return response.content

def generate_quiz_with_solutions(summary, llm):
    messages = [
        SystemMessage(content="당신은 한국 대학생을 위한 예상 문제와 해설을 작성하는 도우미입니다."),
        HumanMessage(content=f"다음 텍스트를 기반으로 중요도를 표시한 예상 문제와 해설을 작성해주세요:\n{summary}")
    ]
    response = llm(messages).content.split("\n\n")
    questions = response[::2]
    solutions = response[1::2]
    return questions, solutions

def evaluate_answers(solutions):
    scores = []
    correct = 0
    for i, solution in enumerate(solutions):
        user_answer = st.session_state.get(f"solution_{i}", "").strip()
        if user_answer.lower() == solution.lower():
            correct += 1
        scores.append(user_answer == solution)
    accuracy = correct / len(solutions)
    st.session_state.evaluation_scores = {"accuracy": accuracy, "total_score": correct}

def visualize_roadmap(roadmap, days_left):
    tasks = roadmap.split("\n")
    y = np.arange(len(tasks))
    plt.figure(figsize=(8, 6))
    plt.barh(y, [1] * len(tasks), color="skyblue")
    plt.yticks(y, tasks)
    plt.xlabel("완료 여부")
    plt.title("공부 로드맵")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
