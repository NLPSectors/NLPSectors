import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def app():
    pkl_filename = "Pickles/all_filenames.pkl"
    with open(pkl_filename, 'rb') as file:
        all_filenames = pickle.load(file)
    all_filenames.append('Todos los posibles')


    pkl_filename = "Pickles/dataset_cl.pkl"
    with open(pkl_filename, 'rb') as file:
        dataset_cl = pickle.load(file)


    pkl_filename = "Pickles/clusterlabels.pkl"
    with open(pkl_filename, 'rb') as file:
        clusterlabels = pickle.load(file)


    data=dataset_cl.drop(['ticker'],axis=1)   


#CREACION DE TOPICS
    def textos_def_topics(textos,etiquetas_topic,ticker):
        #crear un documento único por clase
        docs_df = pd.DataFrame(textos, columns=["Doc"]) #me pongo en un df todos los datos iniciales
        docs_df['Topic'] = etiquetas_topic #Les coloco a cada uno el topic numerico al que pertenecen
        docs_df['Doc_ID'] = range(len(docs_df)) #Les doy el numero que me permite localizarlos
        docs_df['ticker']=ticker
        docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join}) #uno todos segun su topic

        return docs_per_topic,docs_df

    docs_per_topic,docs_df=textos_def_topics(data.values,clusterlabels,dataset_cl.ticker)

    def c_tf_idf(documents, m, ngram_range=(1, 1)):
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count
    
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))


    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def extract_topic_sizes(df):
        topic_sizes = (df.groupby(['Topic'])
                        .Doc
                        .count()
                        .reset_index()
                        .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                        .sort_values("Size", ascending=False))
        return topic_sizes

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df); topic_sizes.sort_values("Topic") #con esto podriamos tener el numero de empresas por sectores nuevos, -1 significa que no está muy biene xplicado. 




    st.title('INVERSIÓN TEMÁTICA')
    st.subheader('Elija el tema que desee, el algoritmo buscará qué sectores coinciden y qué empresas pertenecen a estos sectores')


    palabra=st.text_input("Introduzca la palabra en la que quiere invertir:")

    sector_palabra=[]
    for sector in list(top_n_words.keys()):
        for palabras in range(len(top_n_words[sector])):
            if top_n_words[sector][palabras][0]==palabra.lower():
                sector_palabra.append(sector)
                break

    if len(sector_palabra)==0:
        st.write('No hay ningún sector que hable de ese tema')
    else:
        listado_de_sectores=[]
        st.write('Los sectores existentes con esta palabra son:')
        for sectors in range(len(sector_palabra)):
            st.write(sector_palabra[sectors])
            listado_de_sectores.append(sector_palabra[sectors])
        for columna in range(len(sector_palabra)):
            topic_label=sector_palabra[columna]
            palabras_elegidasv3=[]
            for i in range(len(top_n_words[topic_label][:20])):
                palabras_elegidasv3.append(top_n_words[topic_label][:20][i][0])
            st.write('Las palabras para el sector',sector_palabra[columna],'son:')
            #st.write(sector_palabra[columna])
            st.write(palabras_elegidasv3)

    numero_topicv2=-1
    if len(sector_palabra)>1:
        numero_topicv2 = st.selectbox(
        'Elija sector a analizar:',
        sector_palabra)
    elif len(sector_palabra)!=0:
        numero_topicv2=sector_palabra[0]



    mercado_elegido_compararv2 = st.selectbox(
        'Elija un mercado en el que buscar empresas del mismo grupo:',
        all_filenames)

    if mercado_elegido_compararv2=='Todos los posibles':
        dataset_mercado_compararv2=dataset_cl
    else:
        rutav3=r"TextosCSV/"+mercado_elegido_compararv2
        dataset_mercado_compararv2 = pd.concat([pd.read_csv(rutav3,delimiter= ',',header = 'infer',engine='python')])
        dataset_mercado_compararv2=dataset_mercado_compararv2.iloc[:,1:] #elimino la primera columna del indice, me quito los duplicados y luego reseteo el index
        dataset_mercado_compararv2.columns = ['ticker', 'frases']
        dataset_mercado_compararv2=dataset_mercado_compararv2.drop_duplicates(subset=['ticker']).reset_index(drop=True)
        
    indice_seleccionv2=dataset_cl[dataset_cl.ticker.isin(dataset_mercado_compararv2.ticker)].index



    st.write('Las empresas del mismo Topic que podemos encontrar son:')
    empresas_similaresv2=docs_df.loc[indice_seleccionv2][docs_df.loc[indice_seleccionv2].Topic==numero_topicv2]

    if len(empresas_similaresv2)==0:
        st.write('No hay empresas similares en este mercado:')

    else:
        st.write(empresas_similaresv2)
        definicion_similarcompanyv2= st.selectbox(
            'Elija una empresa para ver la definición',
            empresas_similaresv2.ticker)

        st.write(docs_df.loc[definicion_similarcompanyv2==docs_df.ticker].Doc.values[0])


