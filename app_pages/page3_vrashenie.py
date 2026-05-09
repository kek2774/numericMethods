import streamlit as st

st.set_page_config(
    page_title="Метод вращения для вычисления собственных значений", layout="wide"
)

st.title(
    "Задание 3: методом вращения вычислить собственные значения и собственные вектора симметричной матрицы"
)


n = st.number_input("Введите номер варианта", min_value=1, max_value=20, value=6)

eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)

# do_vrashenie
