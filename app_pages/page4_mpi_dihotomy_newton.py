import streamlit as st

st.set_page_config(
    page_title="МПИ, дихотомия и Ньютон для вычисления корней уравнения", layout="wide"
)

st.title("Задание 4: уточнить один из корней уравнения")

user_func = st.text_input("Введите уравнение", value="x + lg(x) = 0.5")

eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)

methods_seq = ["Метод простой итерации", "Метод дихотомии", "Метод Ньютона"]
method = st.selectbox("Выберите метод", methods_seq, width=240)
eps = st.number_input(
    "Введите точность", min_value=0.000001, max_value=10.0, width=240, value=0.01
)
if method == methods_seq[0]:
    pass
    # do_mpi_func
elif method == methods_seq[1]:
    pass
    # do_dihotomy

elif method == methods_seq[2]:
    pass
    # do_newton
