import streamlit as st

st.set_page_config(page_title="Главная страница расчетки", layout="wide")

pages = [
    st.Page("app_pages/page0_main_page.py", title="Главная страница"),
    st.Page("app_pages/page1_gauss_mpi.py", title="Задание 1: Гаусс, МПИ"),
    st.Page("app_pages/page2_zeidel_progonka.py", title="Задание 2: Зейдель, прогонка"),
    st.Page("app_pages/page3_vrashenie.py", title="123"),
    st.Page("app_pages/page4_mpi_dihotomy_newton.py", title="12345"),
    st.Page("app_pages/page5_lagrange_newton.py", title="Лагранж"),
]
page = st.navigation(pages, position="sidebar")
page.run()
