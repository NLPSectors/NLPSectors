import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

def app():


    pkl_filename = "Pickles/all_filenames.pkl"
    with open(pkl_filename, 'rb') as file:
        all_filenames = pickle.load(file)

    pkl_filename = "Pickles/dataset_cl.pkl"
    with open(pkl_filename, 'rb') as file:
        dataset_cl = pickle.load(file)

    pkl_filename = "Pickles/clusterlabels.pkl"
    with open(pkl_filename, 'rb') as file:
        clusterlabels = pickle.load(file)



    st.title('CLUSTERIZACIÓN INICIAL')
    image = Image.open('Imagenes/clusterizacion.png')
    st.image(image, caption='Clusterizacion en 2D de las empresas analizadas')

    all_filenames.append('Todos los posibles')

    mercado_elegido = st.selectbox(
        '¿En que mercado quieres analizar los NLP Sectors?',
        all_filenames)

    ruta=r"TextosCSV/"+mercado_elegido
    
    data=dataset_cl.drop(['ticker'],axis=1)   
    

    if mercado_elegido=='Todos los posibles':
        dataset_mercado=dataset_cl
    else:

        dataset_mercado = pd.concat([pd.read_csv(ruta,delimiter= ',',header = 'infer',engine='python')])
        dataset_mercado=dataset_mercado.iloc[:,1:] #elimino la primera columna del indice, me quito los duplicados y luego reseteo el index
        dataset_mercado.columns = ['ticker', 'frases']
        dataset_mercado=dataset_mercado.drop_duplicates(subset=['ticker']).reset_index(drop=True)

    indice_seleccion=dataset_cl[dataset_cl.ticker.isin(dataset_mercado.ticker)].index


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


    topic_sizes_mercado = extract_topic_sizes(docs_df.loc[indice_seleccion])
    topic_sizes_mercado=topic_sizes_mercado.sort_values("Topic")

    st.write('El numero de temas en total que hay es de ',len(docs_per_topic), 'pero para el mercado elegido nos encontramos con un número de temas de ', len(topic_sizes_mercado))
    st.write('Para el mercado elegido el número total de empresas es de ', topic_sizes_mercado.Size.sum())
    st.write('Del mercado seleccionado se ha obtenido la siguiente lista de topics')
    st.write(topic_sizes_mercado)

    topic_label = st.selectbox(
        'Elija un topic para ver como es:',
        topic_sizes_mercado.Topic)

    palabras_elegidas=[]
    for i in range(len(top_n_words[topic_label][:20])):
        palabras_elegidas.append(top_n_words[topic_label][:20][i][0])

    st.write(palabras_elegidas)

    ticker_elegido = st.selectbox(
        'Elija una empresa dentro del topic para ver la descripción:',
        docs_df.loc[indice_seleccion][docs_df.Topic==topic_label].ticker.values)

    st.write('La definición de la empresa seleccionada es la siguiente:')
    st.write(docs_df.loc[indice_seleccion][ticker_elegido==docs_df.ticker].Doc.values[0])

