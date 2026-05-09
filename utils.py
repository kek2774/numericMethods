import streamlit as st
import numpy as np


def input_matrix_SLAY(matrix_dim: int):
    result_matrixes = dict()
    A = []
    b = []
    for i in range(matrix_dim):
        cols = st.columns(matrix_dim + 1)
        row = []

        for j in range(matrix_dim):
            with cols[j]:
                st.markdown(rf"$a_{{{i+1}{j+1}}}$")
                value = st.number_input(
                    label=f"a_{i+1}{j+1}",
                    value=None,
                    placeholder="0,00",
                    step=0.1,
                    width=90,
                    label_visibility="collapsed",
                    key=f"a_{i}_{j}",
                )
                row.append(value)

        with cols[-1]:
            st.markdown(rf"$b_{{{i+1}}}$")
            value_b = st.number_input(
                label=f"b_{i+1}",
                value=None,
                placeholder="0,00",
                step=0.1,
                width=90,
                label_visibility="collapsed",
                key=f"b_{i}",
            )
            b.append(value_b)

        A.append(row)

    result_matrixes["A"] = np.array(A, dtype=float)
    result_matrixes["b"] = np.array(b, dtype=float)
    return result_matrixes


def input_matrix(matrix_dim: int):
    A: list = list()
    for i in range(matrix_dim):
        cols = st.columns(matrix_dim + 1)
        row = []

        for j in range(matrix_dim):
            with cols[j]:
                st.markdown(rf"$a_{{{i+1}{j+1}}}$")
                value = st.number_input(
                    label=f"a_{i+1}{j+1}",
                    value=None,
                    placeholder="0,00",
                    step=0.1,
                    width=90,
                    label_visibility="collapsed",
                    key=f"a_{i}_{j}",
                )
                row.append(value)

        A.append(row)

    return np.array(A, dtype=float)
