import streamlit as st

st.set_page_config(page_title="Главная страница расчетки", layout="wide")

pages = [
    st.Page("app_pages/page0_main_page.py", title="Главная страница"),
    st.Page("app_pages/page1_gauss_mpi.py", title="Задание 1: Гаусс, МПИ"),
    st.Page("app_pages/page2_zeidel_progonka.py", title="Задание 2: Зейдель, прогонка"),
]
page = st.navigation(pages, position="sidebar")
page.run()
