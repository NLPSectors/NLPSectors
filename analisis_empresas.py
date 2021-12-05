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



    st.title('ANÁLISIS DE EMPRESAS')
    st.subheader('Elija una empresa, analizaremos al sector al que pertenece y buscaremos empresas similares basadas en el mercado que se desee.')

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

    mercado_elegido_inicio = st.selectbox(
        'Elija un mercado en el que buscar la empresa analizar:',
        all_filenames)

    if mercado_elegido_inicio=='Todos los posibles':
        dataset_mercado_inicio=dataset_cl
    else:
        rutav2=r"TextosCSV/"+mercado_elegido_inicio
        dataset_mercado_inicio = pd.concat([pd.read_csv(rutav2,delimiter= ',',header = 'infer',engine='python')])
        dataset_mercado_inicio=dataset_mercado_inicio.iloc[:,1:] #elimino la primera columna del indice, me quito los duplicados y luego reseteo el index
        dataset_mercado_inicio.columns = ['ticker', 'frases']
        dataset_mercado_inicio=dataset_mercado_inicio.drop_duplicates(subset=['ticker']).reset_index(drop=True)
        
    indice_seleccion=dataset_cl[dataset_cl.ticker.isin(dataset_mercado_inicio.ticker)].index




    empresa_elegido = st.selectbox(
        'Elija una empresa para analizarla:',
         docs_df.loc[indice_seleccion].ticker.values)


    st.write('La definición de la empresa seleccionada es la siguiente:')
    st.write(docs_df[docs_df.ticker==empresa_elegido].Doc.values[0])


    palabras_elegidasv2=[]
    for i in range(len(top_n_words[0][:20])):
        palabras_elegidasv2.append(top_n_words[docs_df[docs_df.ticker==empresa_elegido].Topic.values[0]][:20][i][0])
 

    topic_empresa=docs_df[docs_df.ticker==empresa_elegido].Topic.values[0]
    st.write('Esta empresa pertenece al sector número',topic_empresa, 'y las palabras que definen a este sector son:')
    st.write(palabras_elegidasv2)


    numero_topic=docs_df[docs_df.ticker==empresa_elegido].Topic.values[0]

    mercado_elegido_comparar = st.selectbox(
        'Elija un mercado en el que buscar empresas del mismo topic:',
        all_filenames)

    if mercado_elegido_comparar=='Todos los posibles':
        dataset_mercado_comparar=dataset_cl
    else:
        rutav2=r"TextosCSV/"+mercado_elegido_comparar
        dataset_mercado_comparar = pd.concat([pd.read_csv(rutav2,delimiter= ',',header = 'infer',engine='python')])
        dataset_mercado_comparar=dataset_mercado_comparar.iloc[:,1:] #elimino la primera columna del indice, me quito los duplicados y luego reseteo el index
        dataset_mercado_comparar.columns = ['ticker', 'frases']
        dataset_mercado_comparar=dataset_mercado_comparar.drop_duplicates(subset=['ticker']).reset_index(drop=True)
        
    indice_seleccion=dataset_cl[dataset_cl.ticker.isin(dataset_mercado_comparar.ticker)].index


    st.write('Las empresas del mismo Topic que podemos encontrar son:')
    empresas_similares=docs_df.loc[indice_seleccion][docs_df.loc[indice_seleccion].Topic==numero_topic]
    st.write(empresas_similares)


        

    definicion_similarcompany= st.selectbox(
        'Elija una empresa para ver su definición',
        empresas_similares.ticker)

    st.write(docs_df.loc[definicion_similarcompany==docs_df.ticker].Doc.values[0])


