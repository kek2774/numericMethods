import streamlit as st
from utils import input_matrix_SLAY

st.set_page_config(
    page_title="Метод Зейделя и метод прогонки для решения СЛАУ", layout="wide"
)
st.title("Задание 2: решить СЛАУ методом прогонки и методом Зейделя")
dim = st.number_input("Размерность системы", min_value=2, max_value=9, width=240)
st.markdown(r"Введите коэффициенты системы")
result_matrixes = input_matrix_SLAY(dim)
A = result_matrixes["A"]
b = result_matrixes["b"]

methods_seq = ["Метод прогонки", "Метод Зейделя"]
method = st.selectbox("Выберите метод", methods_seq, width=240)
eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)
if method == methods_seq[0]:
    pass
    # do_progonka
elif method == methods_seq[1]:
    pass
    # do_zeidel
