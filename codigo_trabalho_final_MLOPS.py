import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
# testando_git

#obter dados do dataset

def main():
   dataset_name_treino = "seattle-weather.csv"
   # dataset_name_treino_drift = "fraudTrain_2019_fev.csv"
   # dataset_name_teste = "fraudTest.csv"

   #pegando dados de teste
   df_treino = getData(dataset_name_treino)
   # discover_dataframe(df_treino)

   # df_teste = getData(dataset_name_teste)
   # discover_dataframe(df_teste)

   # x_train,y_train = 

   df_1, df_2_drift = preparacao_dados(df_treino)
   discover_dataframe(df_1)

   # x = df_1.drop("weather")
   x = df_1.drop('weather', axis=1)
   y = df_1['weather']
   # y = df_1["weather"]

   # print("x = ")
   # print(x.head)
   # print("y = ")
   # print(y.head)


   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

   print("x_train = ")
   print(x_train)

   mlflow.set_tracking_uri("sqlite:///mlflow.db")
   mlflow.set_experiment("mlops_intro")

   with mlflow.start_run() as run:
      mlflow.log_param("model_type", "RandomForestClassifier")
      
      model = RandomForestClassifier(n_estimators=50, random_state=42)
      model.fit(x_train, y_train)

      y_pred = model.predict(x_test)
      # rmse = mean_squared_error(y_test, y_pred)
      # mlflow.log_metric("rmse", rmse)

      mlflow.sklearn.log_model(model, "random_forest_classifier_model")
      print(f"Modelo com run Id: {run.info.run_id}")


def treinar_modelo(X, y, dataset_name, mlflow_dataset):
   X_train, X_test, y_train, y_test = train_test_split(X, y,)

def getData(dataset_name):
#   dataframe = pd.read_csv(dataset_name)
  dataframe = pd.read_csv(dataset_name, sep=',')
  discover_dataframe(dataframe)
  return dataframe

def discover_dataframe(df):
   print("tamanho:")
   print(df.size)
   print("colunas:")
   print(df.columns)
   print("head:")
   print(df.head)

   primeira_linha = df.iloc[0]
   print(primeira_linha)

def preparacao_dados(df):

   for col in df.select_dtypes(include=["int64"]).columns:
      df[col] = df[col].astype("float64")

   for col in df.selet_dtypes(include=["object"]).columns:
      df[col] = df[col].astype(str)
      
      #discretizacao
      df[col] = LabelEncoder().fit_transform(df[col])

   #para todos os dados ausentes, colocar valor 0
   df.fillna(0, inPlace=True)

   df_1 = df[df['date'] < '2015-01-01']
   df_2 = df[df['date'] >= '2015-01-01']
   return df_1,df_2




if __name__ == "__main__":
    main()
 