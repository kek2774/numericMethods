import streamlit as st

st.set_page_config(page_title="Интегрирование", layout="wide")

st.title("Задание 8: найти значения 1-й и 2-й производных в точках")

user_integral = st.text_input("Введите уравнение", value="x + lg(x) = 0.5")

eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)
