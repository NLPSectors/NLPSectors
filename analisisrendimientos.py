from os import write
import pandas as pd
import streamlit as st
import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def app():

    pkl_filename = "Pickles/dataset_cl.pkl"
    with open(pkl_filename, 'rb') as file:
        dataset_cl = pickle.load(file)

    pkl_filename = "Pickles/clusterlabels.pkl"
    with open(pkl_filename, 'rb') as file:
        clusterlabels = pickle.load(file)


    etf_old = pd.read_csv('old_etfs/ETFs_close_Vanguard.csv', index_col=0,sep = ';',parse_dates=False)
    new_index = pd.read_csv('creationetfs/Cotizaciones_indices_new.csv', index_col=0,sep = ',')


    st.title('COMPARACIÓN DE LOS RENDIMIENTOS DEL SP500')


    ruta=r"TextosCSV/Descriptions SP500 today and all years_Refinitiv.csv"

    data=dataset_cl.drop(['ticker'],axis=1)   

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

    ayudas=['No necesito ayuda','Buscar en que consiste un tema en particular','Buscar una palabra para ver a que tema pertenece']
    eleccion=st.radio('Si necesita ayuda en saber en que consiste cada sector tiene varias opciones para elegir:',ayudas)

    if eleccion=='Buscar en que consiste un tema en particular':
        
        topic_label = st.selectbox(
            'Elija un topic para ver como es:',
            topic_sizes_mercado.Topic)

        palabras_elegidas=[]
        for i in range(len(top_n_words[topic_label][:20])):
            palabras_elegidas.append(top_n_words[topic_label][:20][i][0])

        st.write(palabras_elegidas)

        ticker_elegido = st.selectbox(
            'Este el listado de las empresas que estan dentros del sector:',
            docs_df.loc[indice_seleccion][docs_df.Topic==topic_label].ticker.values)

        st.write('La definición de la empresa seleccionada es la siguiente:')
        st.write(docs_df.loc[indice_seleccion][ticker_elegido==docs_df.ticker].Doc.values[0])

    elif eleccion=='Buscar una palabra para ver a que tema pertenece':
            
        palabra=st.text_input("Introduzca una palabra para ver que sectores la contienen")

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

    
    col1, col2= st.columns(2)
    with col1:
        tick_etfs=[0]*len(etf_old.columns)
        i=0 
        st.header("Selecciones los ETFs Sectoriales Tradicionales")
        for etfs in etf_old.columns:
            tick_etfs[i] = st.checkbox(etfs)
            i=i+1
    with col2:
        tick_sectors=[0]*len(new_index.columns)
        j=0
        st.header("Seleccione los NLP Sectors")
        
        for nlpsectors in new_index.columns.map(int).sort_values().map(str):
            tick_sectors[j] = st.checkbox(nlpsectors)
            j=j+1

  

    etf_old.columns=['(VCR)','(VDC)','(VDE)','(VFH)','(VHT)','(VIS)','(VGT)','(VAW)','(VNQ)','(VOX)','(VPU)']

    siglas_etf={'Vanguard Consumer Discretion ETF' : '(VCR)',
    'Vanguard Consumer Staples ETF' : '(VDC)',
    'Vanguard Energy ETF' : '(VDE)',
    'Vanguard Financials ETF' : '(VFH)',
    'Vanguard Health Care ETF' : '(VHT)',
    'Vanguard Industrials ETF' : '(VIS)',
    'Vanguard Information Tech ETF'  :'(VGT)',
    'Vanguard Materials ETF' : '(VAW)',
    'Vanguard REIT ETF' : '(VNQ)',
    'Vanguard Communication Services ETF' : '(VOX)',
    'Vanguard Utilities ETF' : '(VPU)'
    }


    vector=etf_old.index.format()
    for i in range(len(vector)):

        vector[i]=datetime.datetime.strptime(vector[i], '%d/%m/%Y').strftime('%y/%m/%d')

    etf_old.index=pd.to_datetime(vector,yearfirst=True)


    etf_old_ret=etf_old.copy()

    etf_old_ret=np.log(etf_old_ret).diff()
    etf_old_ret.iloc[0,:]=1
    etf_old_ret=etf_old_ret.cumsum()



    vector2=new_index.index.format()


    new_index.index=pd.to_datetime(vector2,yearfirst=True)

    new_index_ret=new_index.copy()

    new_index_ret=np.log(new_index_ret).diff()
    new_index_ret.iloc[0,:]=1
    new_index_ret=new_index_ret.cumsum()


    etf_old_seleccion=etf_old.loc[:,tick_etfs].copy()
    etf_old_ret_seleccion=etf_old_ret.loc[:,tick_etfs].copy()
    new_index_seleccion=new_index.loc[:,tick_sectors].copy()
    new_index_ret_seleccion=new_index_ret.loc[:,tick_sectors].copy()



    seleccion = pd.concat([etf_old_seleccion, new_index_seleccion], axis=1)
    seleccion=seleccion.dropna() 
    rent_seleccion=pd.concat([etf_old_ret_seleccion, new_index_ret_seleccion], axis=1)
    rent_seleccion=rent_seleccion.dropna() 


    st.header('Comportamiento:')
    st.line_chart(seleccion,use_container_width=True)
    st.write(seleccion)
    st.header('Rendimientos:    ')
    st.line_chart(rent_seleccion,use_container_width=True)

