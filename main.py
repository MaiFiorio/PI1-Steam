from fastapi import FastAPI, Depends    
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

# uvicorn main:app --reload

#Rutas en GitHug
# df_steam = pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip')
# df_items = pd.read_parquet("./Bases de datos/Archivos Post ETL/items_post_etl.csv", sep=';', on_bad_lines='skip')
# df_reviews = pd.read_csv("./Bases de datos/Archivos Post ETL/reviews_post_etl.csv", sep=';', on_bad_lines='skip')

#Rutas en Local
# df_steam = pd.read_csv(r"M:\Documentos\Mai\Henry\Cursado\P.I. 1\Bases de datos\Archivos Post ETL\steam_post_etl.csv", sep=';', on_bad_lines='skip')
# df_items = pd.read_csv(r"M:\Documentos\Mai\Henry\Cursado\P.I. 1\Bases de datos\Archivos Post ETL\items_post_etl.csv", sep=';', on_bad_lines='skip')
# df_reviews = pd.read_csv(r"M:\Documentos\Mai\Henry\Cursado\P.I. 1\Bases de datos\Archivos Post ETL\reviews_post_etl.csv", sep=';', on_bad_lines='skip')



app = FastAPI(
    title = "Consultas y recomendaciones de VideoJuegos", 
    description = "API que permite realizar consultas sobre VideoJuegos"
)

#________________HOME_________________________
@app.get('/')
async def Home ():
    return "Bienvenidos a la API para Consultas y Recomendaciones de VideoJuegos"
         
#_________PRIMERA FUNCION: developer__________
@app.get('/developer/{desarrollador}')
async def developer( desarrollador : str ):
    """Cantidad de items y porcentaje de contenido Free 
        por año según empresa desarrolladora"""
     
    df_steam = pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['developer', 'year', 'price']) 
        
    #Filtrar juegos por desarrollador
    df_desarrollador = df_steam[df_steam['developer'] == desarrollador].copy()
    
    # Filtrar para eliminar años iguales a 0
    df_desarrollador = df_desarrollador[df_desarrollador['year'] != 0]
    
    # Contar la cantidad de items por año
    cantidad_items_por_year = df_desarrollador.groupby('year').size().reset_index(name="cantidad_items")
    
    # Contar la cantidad de items GRATUITOS por año
    cantidad_items_gratuitos_por_year = df_desarrollador[df_desarrollador["price"] == 0].groupby('year').size().reset_index(name='cantidad_items_gratuitos_por_año').astype('Int64')

    # Combinar filtros para dar resultado y creación de columna que calcule el porcentaje sobre el total
    resultado = pd.merge(cantidad_items_por_year, cantidad_items_gratuitos_por_year, on='year', how='left').fillna(0)
    resultado['% Contenido Free'] = ((resultado['cantidad_items_gratuitos_por_año'] / resultado['cantidad_items']) * 100).round(2)
    
    # Convertir el DataFrame a un diccionario
    resultado_dict = resultado.to_dict(orient='records')
    
    return resultado_dict


#_________SEGUNDA FUNCION: user________________
@app.get('/userdata/{User_id}')
async def userdata( User_id : str ):
    """Devuelve cantidad de dinero gastado por el usuario, el porcentaje de 
    recomendación en base a reviews.recommend y cantidad de items"""

    df_items = pd.read_parquet("./Bases de datos/Archivos Post ETL/items_post_etl.parquet", columns=['user_id', 'item_id', 'price'])
    df_reviews = pd.read_csv("./Bases de datos/Archivos Post ETL/reviews_post_etl.csv", sep=';', on_bad_lines='skip',usecols=['user_id','recommend'])
    
    #Filtrar por user_id
    df_user = df_items[df_items['user_id']== User_id]

    #Calcular el gasto total por  usuario
    gasto = str(round(df_user['price'].sum(),2))+' USD'

    #Calcular la cantidad de items comprados
    cantidad_items = df_user['item_id'].count()
    
    # Filtrar reviews por ususario 
    df_reviews_usuario = df_reviews[df_reviews['user_id']== User_id]
    
    #Calcular la cantidad de juegos recomendados (True)
    cantidad_recomendaciones = (df_reviews_usuario['recommend'] == True).sum()

    # Calcular la cantidad de juegos que tienen datos en la columna review
    cantidad_juegos_con_review = df_reviews_usuario['recommend'].notna().sum()

    #Porcentaje de recomendación
    if cantidad_items > 0:
        porcentaje_recomendaciones = str(round(cantidad_recomendaciones * 100 / cantidad_juegos_con_review, 2)) + '%'
    else:
        porcentaje_recomendaciones = '0%'

    #Muestra los resultados en un diccionario y con el formato adecuado
    return {"Id Usuario" : str(User_id), 
           "Dinero gastado": str(gasto), 
           "% de recomendación": str(porcentaje_recomendaciones),
           "Cantidad de items": int(cantidad_items)  
           }


#_________TERCERA FUNCION: UserForGenre__________
@app.get('/UserForGenre/{genero}')
async def UserForGenre( genero : str ): 
    """Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista 
    de la acumulación de horas jugadas por año de lanzamiento."""
    
    df_items = pd.read_parquet("./Bases de datos/Archivos Post ETL/items_post_etl.parquet", columns=['user_id', 'item_id', 'playtime_forever']) 
    df_steam = pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['item_id', 'year',"genres"])
    
    genero = genero.lower()
    genero = genero.title()

    # Filtrar el DataFrame df_steam para obtener los juegos del género especificado
    df_steam_filtrado  = df_steam[df_steam['genres'].apply(lambda x: genero in x)]
    # Convertir la columna en str
    df_steam_filtrado['item_id'] = df_steam['item_id'].astype(str)
   
    del df_steam
    # Unir df_steam_filtrado con df_items basado en 'item_id' para obtener los usuarios y horas jugadas para los juegos del género 
    df_items.set_index('item_id', inplace=True)
    df_steam_filtrado.set_index('item_id', inplace=True)
    
    df_combinado = df_items.join(df_steam_filtrado[['year']], on='item_id', how='inner')
   
    # DIFERENTES OPCIONES DE PRUBA PARA OPTIMIZACIÓN DE MEMORIA
    # df_combinado = pd.merge(df_items[['user_id', 'playtime_forever']], df_steam_filtrado[['year']], left_index=True, right_index=True, how='inner')
    # df_combinado = pd.merge(df_steam_filtrado[['item_id', 'year',"genres"]], df_items[['user_id', 'item_id', 'playtime_forever']], on='item_id', how='inner')
    
    # Filtrar para eliminar años iguales a 0
    df_combinado = df_combinado[df_combinado['year'] != 0]
    
    del df_steam_filtrado,  df_items
    

    #Obtener el usuario con más horas jugadas y luego agrupar  por año
    usuario_mas_horas = df_combinado.groupby('user_id')['playtime_forever'].sum().idxmax()
    horas_por_año_usuario = df_combinado[df_combinado['user_id'] == usuario_mas_horas].groupby('year')['playtime_forever'].sum().reset_index()
    del df_combinado
    
    # Convertir a la lista deseada de acumulación de horas jugadas por año
    dicc_horas_por_year = [{'Año': row['year'], 'Horas': row['playtime_forever']} for index, row in horas_por_año_usuario.iterrows()]

    respuesta = {
    f"Usuario con más horas jugadas para Género {genero}": usuario_mas_horas,
    "Horas jugadas": dicc_horas_por_year }

    return respuesta


#________CUARTA FUNCION: best_developer_year_________
@app.get('/best_developer_year/{year}')
async def best_developer_year( year : int ):
    """Devuelve el top 3 de desarrolladores con juegos 
    MÁS recomendados por usuarios para el año dado. """
    
    df_steam = pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['item_id', 'year',"developer"])
    df_reviews = pd.read_csv("./Bases de datos/Archivos Post ETL/reviews_post_etl.csv", sep=';', on_bad_lines='skip',usecols=['user_id','item_id','recommend'])
    
    # Filtrar df_steam por el año dado
    juegos_por_año = df_steam[df_steam['year'] == year]

    # Unir df_steam con df_reviews por item_id
    merged_df = pd.merge(juegos_por_año, df_reviews, on='item_id', how='left')

    # Filtrar solo las recomendaciones positivas
    recomendaciones_positivas = merged_df[merged_df['recommend'] == True]

    # Contar recomendaciones por desarrollador
    conteo_recomendaciones = recomendaciones_positivas.groupby('developer').size().reset_index(name='cantidad_recomendaciones')

    # Ordenar df para obtener los primeros 3 desarrolladores segun la cantidad de recomendaciones
    top_desarrolladores = conteo_recomendaciones.sort_values(by='cantidad_recomendaciones', ascending=False).head(3)

    # Guardo los tres primeros puestos
    primer_puesto = top_desarrolladores.iloc[0]['developer']
    segundo_puesto = top_desarrolladores.iloc[1]['developer']
    tercer_puesto = top_desarrolladores.iloc[2]['developer']

    # Formatear el resultado
    resultado = {
        "Puesto 1": primer_puesto,
        "Puesto 2": segundo_puesto,
        "Puesto 3": tercer_puesto
    }
    return resultado


#________QUINTA FUNCION: developer_reviews_analysis_________
@app.get('/developer_reviews_analysis/{desarrolladora}')
async def developer_reviews_analysis( desarrolladora : str ): 
    """Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y 
    una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados 
    con un análisis de sentimiento como valor positivo o negativo."""
    
    df_steam = pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['item_id',"developer"])
    df_reviews = pd.read_csv("./Bases de datos/Archivos Post ETL/reviews_post_etl.csv", sep=';', on_bad_lines='skip',usecols=['user_id','item_id','sentiment_analysis'])
    
    
    # Filtrar df_steam por la desarrolladora especificada
    juegos_desarrolladora = df_steam[df_steam['developer'] == desarrolladora]
    
    # Unir df_steam con df_reviews por item_id
    merged_df = pd.merge(juegos_desarrolladora, df_reviews, on='item_id', how='left')

    # Contar la cantidad total de reseñas positivas y negativas
    conteo_reseñas = merged_df['sentiment_analysis'].value_counts().to_dict()

        # Crear el diccionario de resultados
    reseñas = {
        desarrolladora: [
            conteo_reseñas.get(2.0, 0),   # Positive count
            conteo_reseñas.get(0.0, 0)      # Negative count
        ]
    }

    # Renombrar claves en la lista a 'Positivas' y 'Negativas'
    reseñas[desarrolladora] = {
        "Positivas": reseñas[desarrolladora][0],
        "Negativas": reseñas[desarrolladora][1]
    }

    return reseñas

#________MODELO DE ML: GENERO_________
@app.get('/sistema_recomendacion_por_genero/{item_id}')
async def recomendar_juegos_genero(item_id : int):
    """ Se consulta  un item_id, y se recomiendan 5 juegos basados 
    en su similitud respecto a sus generos respectivos"""
    try:
        # Cargar los dataframes
        df_steam =  pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['item_id', 'title', 'genres']) 
        # df_reviews = pd.read_csv(r"M:\Documentos\Mai\Henry\Cursado\P.I. 1\Bases de datos\Archivos Post ETL\reviews_post_etl.csv", sep=';', usecols=['item_id', 'sentiment_analysis'])
       
        # Verificar si el item_id existe en el dataframe
        if item_id not in df_steam['item_id'].values:
            raise ValueError(f"El item_id {item_id} no está en la base de datos.")

        # Filtrar el dataframe de Steam para obtener solo los juegos relevantes
        df_steam = df_steam.dropna(subset=['genres'])  # Eliminar filas con NaN en géneros

        # Obtener el género del juego buscado
        game_genres_str = df_steam[df_steam['item_id'] == item_id]['genres'].values[0]
        game_genres = ast.literal_eval(game_genres_str)
        
        # verificar si el juego no tiene géneros válidos
        if not game_genres_str or game_genres_str == '[]':
            raise ValueError(f"El juego con item_id {item_id} no tiene géneros asignados, por lo que no podemos darte una recomendación para géneros similares.")


        # Filtro los juegos que tengan al menos alguno de esos generos y restablecer indices
        filtered_df_steam = df_steam[df_steam['genres'].apply(lambda x: any(genre in eval(x) for genre in game_genres))]
        filtered_df_steam = filtered_df_steam.reset_index(drop=True)

        # Si no hay suficientes juegos similares en el filtro
        if filtered_df_steam.shape[0] <= 1:
            raise ValueError(f"No se encontraron suficientes juegos similares al juego con item_id {item_id}.")

        # Vectorizar la columna 'genres':
        # Obj: Convertir el texto de la columna genres en una representación numérica que pueda ser utilizada para calcular similitudes.
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(filtered_df_steam['genres'])
        
         # Obtener el índice del juego buscado
        idx = filtered_df_steam.index[filtered_df_steam['item_id'] == item_id].tolist()[0]

        # Calcular la similitud coseno entre el juego buscado y todos los juegos
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
   

        #Calcular las puntuaciones de similitud entre el juego buscado y todos los demás juegos    
        sim_scores_idx = list(enumerate(sim_scores))

        # Ordenar las puntuaciones de similitud en orden descendente (juegos más similares primero)
        sim_scores_idx = sorted(sim_scores_idx, key=lambda x: x[1], reverse=True)

        # Obtener los índices de los juegos más similares (excluyendo el propio juego)
        # Esto se hace porque la primera entrada (sim_scores[0]) es el propio juego, que tendrá una similitud de 1  y no lo queremos en las recomendaciones.
        top_indices = [i[0] for i in sim_scores_idx[1:6]]

        # Retornar los títulos de los juegos recomendados
        return filtered_df_steam.iloc[top_indices][['title', 'genres']]
    except ValueError as e:
        return str(e)  # Retornar el mensaje de error si ocurre

#________MODELO DE ML: GENERO y ESPECIFICACIONES_________
@app.get('/sistema_recomendacion_por_genero_specs/{item_id}')
async def recomendar_juegos_genero_spec(item_id : int):
    """ Se consulta  un item_id, y se recomiendan 5 juegos basados 
        en su similitud respecto a sus generos y especificaciones"""
    # Cargar el dataframe
    df_steam =  pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['item_id', 'title', 'genres']) 
        
    # Verificar si el item_id existe en el dataframe, sino error
    if item_id not in df_steam['item_id'].values:
        raise ValueError(f"El item_id {item_id} no está en la base de datos.")

    # Filtra el DataFrame para eliminar filas que tengan valores nulos en las columnas genres o specs, y reinicia los índices
    df_steam = df_steam.dropna(subset=['genres', 'specs']).reset_index(drop=True)

    # Selecciona la fila del DataFrame que corresponde al item_id buscado en la funcion
    game_row = df_steam[df_steam['item_id'] == item_id]

    # Verificar si se encontró el juego
    if game_row.empty:
        raise ValueError(f"No se encontró el juego con item_id {item_id}.")

    # Extraer los géneros y especificaciones del juego encontrado
    game_genres_str = game_row['genres'].values[0]
    game_specs_str = game_row['specs'].values[0]
    game_title = game_row['title'].values[0]
    
    #Convierte las cadenas de texto de game_genres_str y game_specs_str
    game_genres = ast.literal_eval(game_genres_str) if isinstance(game_genres_str, str) else game_genres_str
    game_specs = ast.literal_eval(game_specs_str) if isinstance(game_specs_str, str) else game_specs_str

    # Unir en una sola fila Genero + Specs
    genres_specs_item = list(set(game_specs + game_genres))

    # Definir una función interna para Convertir las columnas de strings en listas
    def combinar_generos_specs(row):
        
        genres = ast.literal_eval(row['genres']) if isinstance(row['genres'], str) else row['genres']
        specs = ast.literal_eval(row['specs']) if isinstance(row['specs'], str) else row['specs']
        
        # Unir ambos en una sola lista y eliminar duplicados si los hubiera
        return list(set(genres + specs))

    # Aplicar la función a cada fila del dataframe y crear la nueva columna 'genres_mas_specs'
    df_steam['genres_mas_specs'] = df_steam.apply(combinar_generos_specs, axis=1)

    # Convertir cada lista en 'genres_mas_specs_str' en una cadena de texto
    df_steam['genres_mas_specs_str'] = df_steam['genres_mas_specs'].apply(lambda x: ' '.join(x))

    # Vectorizar la columna 'genres_mas_specs_str'
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_steam['genres_mas_specs_str'])

    # Obtener el índice del juego buscado en el DataFrame
    idx = df_steam.index[df_steam['item_id'] == item_id].tolist()[0]

    # Calcular la similitud coseno entre el juego buscado y todos los juegos
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Calcular las puntuaciones de similitud entre el juego buscado y todos los demás juegos    
    sim_scores_idx = list(enumerate(sim_scores))
    # Ordenar las puntuaciones de similitud en orden descendente (juegos más similares primero)
    sim_scores_idx = sorted(sim_scores_idx, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los juegos más similares (excluyendo el propio juego)
    top_indices = [i[0] for i in sim_scores_idx[1:6]]

    # Crear la respuesta estructurada
    recomendaciones = df_steam.iloc[top_indices][['title', 'genres', 'specs']].to_dict(orient='records')
    
    respuesta = {
        'Juego Buscado': game_title,
        'Top 5 Recomendaciones': recomendaciones
    }
    
    return respuesta