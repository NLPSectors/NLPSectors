import analisis_empresas
import introduccion
import inversion_por_topic
import analisisrendimientos

import streamlit as st

PAGES = {
    "Clusterización inicial": introduccion,
    "Análisis de empresas": analisis_empresas,
    "Inversión temática": inversion_por_topic,
    'Análisis de rendimientos':analisisrendimientos

}
st.sidebar.title('NLP SECTORS')
selection = st.sidebar.radio("Ir a:", list(PAGES.keys()))
page = PAGES[selection]
page.app()