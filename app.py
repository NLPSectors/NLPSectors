import analisis_empresas
import introduccion
import inversion_por_topic

import streamlit as st

PAGES = {
    "Clusterización inicial": introduccion,
    "Análisis de empresas": analisis_empresas,
    "Inversión temática": inversion_por_topic

}
st.sidebar.title('NLP SECTORS')
selection = st.sidebar.radio("Ir a:", list(PAGES.keys()))
page = PAGES[selection]
page.app()