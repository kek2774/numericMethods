import streamlit as st
from utils import input_table

st.set_page_config(page_title="Интерполяционные многочлены", layout="wide")

st.title(
    "Задание 5: выписать интерполяционные многочлены Лагранжа и Ньютона для заданных узловых значений и найти погрешность в точке"
)

x_star = st.number_input("x* = ", value=1.4179)

table_length = st.number_input(
    "Введите число наблюдений", min_value=2, max_value=20, value=6
)
table = input_table(table_length=table_length)

methods_seq = ["Многочлены Лагранжа", "Многочлены Ньютона"]
method = st.selectbox("Выберите многочлен", methods_seq, width=240)

if method == methods_seq[0]:
    pass
    # do_newton
elif method == methods_seq[1]:
    pass
    # do_lagrange
