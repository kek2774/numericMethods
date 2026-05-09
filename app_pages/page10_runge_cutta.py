import streamlit as st

st.set_page_config(page_title="Решение задачи Коши", layout="wide")

st.title("Задание 10: Решить задачу Коши")

user_eq = st.text_input("Введите уравнение", value="x + lg(x) = 0.5")
initial_condition = st.text_input("Введите начальное условие", value="y(0) = 0")

eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)

h_step = st.number_input(
    "Введите шаг", min_value=0.000001, max_value=10.0, width=240, value=0.1
)


methods_seq = ["Метод Рунге-Кутта 4-го порядка", "Метод Адамса"]
method = st.selectbox("Выберите метод", methods_seq, width=240)
eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)
if method == methods_seq[0]:
    pass
elif method == methods_seq[1]:
    pass
