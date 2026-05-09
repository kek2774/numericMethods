import streamlit as st

st.set_page_config(page_title="Решение задачи Коши", layout="wide")

st.title("Задание 11: решить краевую задачу для ОДУ")

user_eq = st.text_input("Введите уравнение", value="y'' - 3y' + y/x = 1")
initial_condition = st.text_input(
    "Введите краевые условия через запятую",
    value="y(0.4) = 2, y(0.7) + 2*y'(0.7) = 0.7",
)

eps = st.text_input("Введите точность", value="O(h^2)")

h_step = st.number_input(
    "Введите шаг", min_value=0.000001, max_value=10.0, width=240, value=0.1
)
