import pandas as pd
from sklearn.model_selection import train_test_split

# testando_git

#obter dados do dataset

def main():
   dataset_name_treino = "fraudTrain_2019_jan.csv"
   dataset_name_treino_drift = "fraudTrain_2019_fev.csv"

   # dataset_name_teste = "fraudTest.csv"

   #pegando dados de teste
   df_treino = getData(dataset_name_treino)
   discover_dataframe(df_treino)

   # df_teste = getData(dataset_name_teste)
   # discover_dataframe(df_teste)

   # x_train,y_train = 

def discover_dataframe(df):
   print("tamanho:")
   print(df.size)
   print("colunas:")
   print(df.columns)
   print("head:")
   print(df.head)

   primeira_linha = df.iloc[1]
   print(primeira_linha)

def getData(dataset_name):
#   dataframe = pd.read_csv(dataset_name)
  dataframe = pd.read_csv(dataset_name, sep=',')
  discover_dataframe(dataframe)
  return dataframe

if __name__ == "__main__":
    main()
 