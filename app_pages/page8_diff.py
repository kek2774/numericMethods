import streamlit as st
from utils import input_table

st.set_page_config(page_title="Значения производных", layout="wide")

st.title("Задание 8: найти значения 1-й и 2-й производных в точках")

x_star = st.number_input("x* = ", value=1.4179)

table_length = st.number_input(
    "Введите число наблюдений", min_value=2, max_value=20, value=6
)
table = input_table(table_length=table_length)
