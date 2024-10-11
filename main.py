from fastapi import FastAPI, Depends
import pandas as pd
import numpy as np
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
    return "Bienvenidos a la API para Consultas y recomendaciones de VideoJuegos"
         
#_________PRIMERA FUNCION: developer__________
@app.get('/developer/{desarrollador}')
async def developer( desarrollador : str ):
    """Cantidad de items y porcentaje de contenido Free 
        por año según empresa desarrolladora"""
     
    df_steam = pd.read_csv("./Bases de datos/Archivos Post ETL/steam_post_etl.csv", sep=';', on_bad_lines='skip', usecols=['developer', 'year', 'price']) 
        
    #Filtrar juegos por desarrollador
    df_desarrollador = df_steam[df_steam['developer'] == desarrollador].copy()
    
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

    # Unir df_steam_filtrado con df_items basado en 'item_id' para obtener los usuarios y horas jugadas para los juegos del género
    df_combinado = pd.merge(df_steam_filtrado[['item_id', 'year',"genres"]], df_items[['user_id', 'item_id', 'playtime_forever']], on='item_id', how='inner')

    # Agrupar por 'user_id' y sumar las horas jugadas para cada usuario
    horas_por_usuario = df_combinado.groupby('user_id')['playtime_forever'].sum().reset_index()

    # Obtener el usuario con más horas jugadas
    usuario_mas_horas = horas_por_usuario.loc[horas_por_usuario['playtime_forever'].idxmax(), 'user_id']

    # Agrupar por año y sumar las horas jugadas para cada año
    horas_por_año_usuario = df_combinado[df_combinado['user_id'] == usuario_mas_horas].groupby('year')['playtime_forever'].sum().reset_index()

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
    