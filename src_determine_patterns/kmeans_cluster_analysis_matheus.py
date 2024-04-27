import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF
import sys 
import glob 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#database_energy_by_periods 
# C:/Users/matheus/Desktop/danilo/energetic_patterns_cyclones_south_atlantic/database_energy_by_periods/
ENERGETICSPATH = 'C:/Users/matheus/Desktop/danilo/energetic_patterns_cyclones_south_atlantic/csv_database_energy_by_periods/'

FIGSAVEPATH = os.path.join('C:/Users/matheus/Desktop/danilo/energetic_patterns_cyclones_south_atlantic/figures/patterns/')


CSVSAVEPATH = os.path.join('C:/Users/matheus/Desktop/danilo/energetic_patterns_cyclones_south_atlantic/csv_patterns/')

all_files = []
files = glob.glob(os.path.join(ENERGETICSPATH, "*.csv"))
all_files.extend(files)

# Creating a list to save all dataframes
cyclist1= []

# Reading all files and saving in a list of dataframes
for case in all_files:
  columns_to_read = ['Ck','Ca', 'Ke', 'Ge']
  dfcyc = pd.read_csv(case,header=0,index_col=0)
  dfcyc = dfcyc[columns_to_read]
  cyclist1.append(dfcyc)

# Inicializando as listas para cada tipo
tipo_1_list = []
tipo_2_list = []
tipo_3_list = []
tipo_4_list = []
tipo_5_list = []

# Definindo os conjuntos de fases para cada tipo
tipo_1 = {'incipient', 'intensification', 'mature', 'decay'}
tipo_2 = {'intensification', 'decay'}
tipo_3 = {'intensification', 'mature', 'decay'}
tipo_4 = {'incipient', 'intensification', 'mature', 'decay', 'residual'}
tipo_5 = {'incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2'}





# Função para classificar e adicionar cada DataFrame à lista correspondente
def classificar_e_adicionar(df):
    fases_presentes = set(df.index.dropna())
    if fases_presentes == tipo_1:
        tipo_1_list.append(df)
    elif fases_presentes == tipo_2:
        tipo_2_list.append(df)
    elif fases_presentes == tipo_3:
        tipo_3_list.append(df)
    elif fases_presentes == tipo_4:
        tipo_4_list.append(df)
    elif fases_presentes == tipo_5:
        tipo_5_list.append(df)

# Aplicando a função de classificação em todos os DataFrames
for df in cyclist1:
    classificar_e_adicionar(df)




def prepare_to_kmeans(cyclist1):

    combined_df = pd.concat(cyclist1, axis=1)

    Ck1 = combined_df['Ck'].values.T
    Ca1 = combined_df['Ca'].values.T
    Ke1	= combined_df['Ke'].values.T
    Ge1 = combined_df['Ge'].values.T

    dsk_means = np.concatenate((Ck1,Ca1,Ke1,Ge1),axis=1)
    
    return dsk_means


dsk_means1 = prepare_to_kmeans(tipo_1_list)
dsk_means2 = prepare_to_kmeans(tipo_2_list)
dsk_means3 = prepare_to_kmeans(tipo_3_list)
dsk_means4 = prepare_to_kmeans(tipo_4_list)
dsk_means5 = prepare_to_kmeans(tipo_5_list)

mk1 = KMeans(n_clusters=4,n_init=10).fit(dsk_means1)
mk2 = KMeans(n_clusters=4,n_init=10).fit(dsk_means2)
mk3 = KMeans(n_clusters=4,n_init=10).fit(dsk_means3)
mk4 = KMeans(n_clusters=4,n_init=10).fit(dsk_means4)
mk5 = KMeans(n_clusters=4,n_init=10).fit(dsk_means5)

def slice_mk(mk, tipo):
    slcenter = len(tipo)
    centers_Ck = mk.cluster_centers_[:,0:slcenter]
    centers_Ca = mk.cluster_centers_[:,slcenter:slcenter*2]
    centers_Ke = mk.cluster_centers_[:,slcenter*2:slcenter*3]
    centers_Ge = mk.cluster_centers_[:,slcenter*3:slcenter*4]
    return centers_Ck, centers_Ca, centers_Ke, centers_Ge

centers_Ck1, centers_Ca1, centers_Ke1, centers_Ge1 = slice_mk(mk1, tipo_1)
centers_Ck2, centers_Ca2, centers_Ke2, centers_Ge2 = slice_mk(mk2, tipo_2)
centers_Ck3, centers_Ca3, centers_Ke3, centers_Ge3 = slice_mk(mk3, tipo_3)
centers_Ck4, centers_Ca4, centers_Ke4, centers_Ge4 = slice_mk(mk4, tipo_4)
centers_Ck5, centers_Ca5, centers_Ke5, centers_Ge5 = slice_mk(mk5, tipo_5)


def sel_clusters_to_df(centers_Ck, centers_Ca, centers_Ke, centers_Ge, tipo_list):
    cl1Ck = centers_Ck[0,:]
    cl2Ck = centers_Ck[1,:]
    cl3Ck = centers_Ck[2,:]
    cl4Ck = centers_Ck[3,:]

    cl1Ca = centers_Ca[0,:]
    cl2Ca = centers_Ca[1,:]
    cl3Ca = centers_Ca[2,:]
    cl4Ca = centers_Ca[3,:]

    cl1Ke = centers_Ke[0,:]
    cl2Ke = centers_Ke[1,:]
    cl3Ke = centers_Ke[2,:]
    cl4Ke = centers_Ke[3,:]

    cl1Ge = centers_Ge[0,:]
    cl2Ge = centers_Ge[1,:]
    cl3Ge = centers_Ge[2,:]
    cl4Ge = centers_Ge[3,:]

    df_cl1 = tipo_list[0].copy()
    df_cl1[:] = np.nan
    df_cl2 = df_cl1.copy()
    df_cl3 = df_cl1.copy()
    df_cl4 = df_cl1.copy()
        #att branch
    df_cl1['Ck'] = cl1Ck
    df_cl1['Ca'] = cl1Ca
    df_cl1['Ke'] = cl1Ke
    df_cl1['Ge'] = cl1Ge

    df_cl2['Ck'] = cl2Ck
    df_cl2['Ca'] = cl2Ca
    df_cl2['Ke'] = cl2Ke
    df_cl2['Ge'] = cl2Ge

    df_cl3['Ck'] = cl3Ck
    df_cl3['Ca'] = cl3Ca
    df_cl3['Ke'] = cl3Ke
    df_cl3['Ge'] = cl3Ge

    df_cl4['Ck'] = cl4Ck
    df_cl4['Ca'] = cl4Ca
    df_cl4['Ke'] = cl4Ke
    df_cl4['Ge'] = cl4Ge

    return df_cl1, df_cl2, df_cl3, df_cl4

df1_cl1, df1_cl2, df1_cl3, df1_cl4 = sel_clusters_to_df(centers_Ck1, centers_Ca1, centers_Ke1, centers_Ge1, tipo_1_list)
df2_cl1, df2_cl2, df2_cl3, df2_cl4 = sel_clusters_to_df(centers_Ck2, centers_Ca2, centers_Ke2, centers_Ge2, tipo_2_list)
df3_cl1, df3_cl2, df3_cl3, df3_cl4 = sel_clusters_to_df(centers_Ck3, centers_Ca3, centers_Ke3, centers_Ge3, tipo_3_list)
df4_cl1, df4_cl2, df4_cl3, df4_cl4 = sel_clusters_to_df(centers_Ck4, centers_Ca4, centers_Ke4, centers_Ge4, tipo_4_list)
df5_cl1, df5_cl2, df5_cl3, df5_cl4 = sel_clusters_to_df(centers_Ck5, centers_Ca5, centers_Ke5, centers_Ge5, tipo_5_list)

# create paste inside CSVSAVEPATH with the tipo name 
# and save the dataframes to csv files


t1_folder = CSVSAVEPATH + 'IcItMD/' 
t2_folder = CSVSAVEPATH + 'ItD/'
t3_folder = CSVSAVEPATH + 'ItMD/'
t4_folder = CSVSAVEPATH + 'IcItMDR/'
t5_folder = CSVSAVEPATH + 'IcItMDIt2M2D2/'

# create these folders 
if not os.path.exists(t1_folder):
    os.makedirs(t1_folder)
if not os.path.exists(t2_folder):
    os.makedirs(t2_folder)
if not os.path.exists(t3_folder):
    os.makedirs(t3_folder)
if not os.path.exists(t4_folder):
    os.makedirs(t4_folder)
if not os.path.exists(t5_folder):
    os.makedirs(t5_folder)


df1_cl1.to_csv(t1_folder + 'df1_cl1.csv')
df1_cl2.to_csv(t1_folder + 'df1_cl2.csv')
df1_cl3.to_csv(t1_folder + 'df1_cl3.csv')
df1_cl4.to_csv(t1_folder + 'df1_cl4.csv')
df2_cl1.to_csv(t2_folder + 'df2_cl1.csv')
df2_cl2.to_csv(t2_folder + 'df2_cl2.csv')
df2_cl3.to_csv(t2_folder + 'df2_cl3.csv')
df2_cl4.to_csv(t2_folder + 'df2_cl4.csv')
df3_cl1.to_csv(t3_folder + 'df3_cl1.csv')
df3_cl2.to_csv(t3_folder + 'df3_cl2.csv')
df3_cl3.to_csv(t3_folder + 'df3_cl3.csv')
df3_cl4.to_csv(t3_folder + 'df3_cl4.csv')
df4_cl1.to_csv(t4_folder + 'df4_cl1.csv')
df4_cl2.to_csv(t4_folder + 'df4_cl2.csv')
df4_cl3.to_csv(t4_folder + 'df4_cl3.csv')
df4_cl4.to_csv(t4_folder + 'df4_cl4.csv')
df5_cl1.to_csv(t5_folder + 'df5_cl1.csv')
df5_cl2.to_csv(t5_folder + 'df5_cl2.csv')
df5_cl3.to_csv(t5_folder + 'df5_cl3.csv')
df5_cl4.to_csv(t5_folder + 'df5_cl4.csv')

