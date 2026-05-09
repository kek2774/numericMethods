import streamlit as st

pages = [
    st.Page("app_pages/page0_main_page.py", title="Главная страница"),
    st.Page("app_pages/page1_gauss_mpi.py", title="Задание 1"),
    st.Page("app_pages/page2_zeidel_progonka.py", title="Задание 2"),
    st.Page("app_pages/page3_vrashenie.py", title="Задание 3"),
    st.Page("app_pages/page4_mpi_dihotomy_newton.py", title="Задание 4"),
    st.Page("app_pages/page5_lagrange_newton.py", title="Задание 5"),
    st.Page("app_pages/page6_cube_spline.py", title="Задание 6"),
    st.Page("app_pages/page7_mls.py", title="Задание 7"),
    st.Page("app_pages/page8_diff.py", title="Задание 8"),
    st.Page("app_pages/page9_simpson.py", title="Задание 9"),
    st.Page("app_pages/page10_runge_cutta.py", title="Задание 10"),
    st.Page("app_pages/page11_progonka.py", title="Задание 11"),
]
page = st.navigation(pages, position="sidebar")
page.run()
