import pickle
import torch.utils as utils
import torch
import pandas as pd

def extract_data(cognitive_variables):
    data_real = pickle.load(
        open(
            '/data/parietal/store2/work/ggomezji/graph_dmri/data/subjects_LH_non_neighbours_mid_ribbon_concatenated.pkl',
            'rb'
        )
    )
    attenuations, cognition = list(zip(*[
        (d.x.squeeze(), torch.Tensor([d[c] for c in cognitive_variables]))
        for d in data_real
    ]))
    attenuations = torch.nan_to_num(torch.stack(attenuations).to(torch.float32))
    cognition = torch.nan_to_num(torch.stack(cognition).to(torch.float32))

    return attenuations, cognition

def real_data():
    cognitive_variables = [
        'Age_in_Yrs', 
        'WM_Task_Acc', 
        'WM_Task_Median_RT',
        'Relational_Task_Acc',
        'Relational_Task_Median_RT', 
        'Gambling_Task_Perc_Larger',
        'Gambling_Task_Median_RT_Larger',
        'ListSort_AgeAdj', 
        'Flanker_AgeAdj',
        'CardSort_AgeAdj',
        'PicSeq_AgeAdj',
        'ProcSpeed_AgeAdj'
    ]
    attenuations, cognition = extract_data(cognitive_variables)
    subject_data = utils.data.TensorDataset(attenuations, cognition)
    train_set, validation_set = utils.data.random_split(
        subject_data, [.9, .1]
    )

    X_train = torch.stack([item[0] for item in train_set])
    Y_train = torch.stack([item[1] for item in train_set])
    X_test = torch.stack([item[0] for item in validation_set])
    Y_test = torch.stack([item[1] for item in validation_set])

    return X_train, Y_train, X_test, Y_test

def tensor_to_dataframe(tensor, columns):
    df = pd.DataFrame(tensor.numpy(), columns=columns)
    return df


X_train, Y_train, X_test, Y_test = real_data()

print (X_train.shape, Y_train.shape)


print(X_test.shape, Y_test.shape)


# Variables cognitivas
cognitive_variables = [
    'Age_in_Yrs', 
    'WM_Task_Acc', 
    'WM_Task_Median_RT',
    'Relational_Task_Acc',
    'Relational_Task_Median_RT', 
    'Gambling_Task_Perc_Larger',
    'Gambling_Task_Median_RT_Larger',
    'ListSort_AgeAdj', 
    'Flanker_AgeAdj',
    'CardSort_AgeAdj',
    'PicSeq_AgeAdj',
    'ProcSpeed_AgeAdj'
]

# Convertir el tensor de Y_train a DataFrame
df_cognition = tensor_to_dataframe(Y_train, cognitive_variables)

# Mostrar las primeras filas y estadísticas descriptivas
print("Primeras filas del DataFrame de Cognición:")
print(df_cognition.head())
print("\nEstadísticas descriptivas del DataFrame de Cognición:")
print(df_cognition.describe())


import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_distributions(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_columns = df.columns
    for column in num_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribución de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        #plt.savefig(os.path.join(output_dir, f'distribucion_{column}.png'))
        #plt.close()

# Visualizar la distribución de los datos de cognición
output_dir = './'
plot_distributions(df_cognition, output_dir)


def detect_outliers(df):
    outliers = pd.DataFrame()
    num_columns = df.columns
    for column in num_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = df[column].apply(lambda x: x < lower_bound or x > upper_bound)
    
    return outliers

# Detectar valores atípicos en los datos de cognición
outliers = detect_outliers(df_cognition)
print("\nValores atípicos detectados (True indica un valor atípico):")
print(outliers)

# Resumen de los valores atípicos
print("\nResumen de valores atípicos:")
print(outliers.sum())


def plot_boxplots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_columns = df.columns
    for column in num_columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot de {column}')
        plt.xlabel(column)
        #plt.savefig(os.path.join(output_dir, f'boxplot_{column}.png'))
        #plt.close()

# Visualizar boxplots para detectar visualmente los valores atípicos
output_dir = './'
plot_boxplots(df_cognition, output_dir)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


scaler = MinMaxScaler()


df_cognition = tensor_to_dataframe

df_cognition_train = tensor_to_dataframe(Y_train, cognitive_variables)
df_cognition_test = tensor_to_dataframe(Y_test, cognitive_variables)

# Ajustar el escalador a los datos de entrenamiento y transformarlos
Y_train_scaled = scaler.fit_transform(df_cognition_train)
Y_test_scaled = scaler.transform(df_cognition_test)

# Convertir de nuevo a tensores
Y_train_scaled = torch.tensor(Y_train_scaled, dtype=torch.float32)
Y_test_scaled = torch.tensor(Y_test_scaled, dtype=torch.float32)

# Reemplazar los datos originales con los datos escalados
Y_train = Y_train_scaled
Y_test = Y_test_scaled

# Mostrar datos normalizados
print("\nDatos de entrenamiento escalados:")
print(Y_train)
print("\nDatos de prueba escalados:")
print(Y_test)


def plot_distributions_and_save(df, output_dir='normalized'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    num_columns = df.columns
    for column in num_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribución de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(output_dir, f'distribucion_{column}.png'))
        plt.close()

df_cognition_train = tensor_to_dataframe(Y_train, cognitive_variables)
plot_distributions_and_save(df_cognition_train)