import mlflow.data.pandas_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.data import pandas_dataset

# testando_git

#obter dados do dataset

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("trabalho_final_experiment_3")

def main():
   print("iniciou main")
   dataset_name_treino = "seattle-weather.csv"
   # dataset_name_treino_drift = "fraudTrain_2019_fev.csv"
   # dataset_name_teste = "fraudTest.csv"

   #pegando dados de teste
   df_treino = getData(dataset_name_treino)
   # descreve_dataframe(df_treino)

   # df_teste = getData(dataset_name_teste)
   # descreve_dataframe(df_teste)

   # x_train,y_train = 
   print("2")

   df_1, df_2_drift,mlflow_dataset = preparacao_dados(df_treino)
   print("df_1 -----------------------------------------------")
   descreve_dataframe(df_1)

   # x = df_1.drop("weather")
   X = df_1.drop('weather', axis=1)
   y = df_1['weather']
   # y = df_1["weather"]

   modelo = treinar_modelo(X, y, dataset_name_treino, mlflow_dataset)
   # print("x = ")
   # print(x.head)
   # print("y = ")
   # print(y.head)


   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # print("X_train = ")
   # print(X_train)

   # mlflow.set_tracking_uri("sqlite:///mlflow.db")
   # mlflow.set_experiment("mlops_intro")

   # with mlflow.start_run() as run:
   #    mlflow.log_param("model_type", "RandomForestClassifier")
      
   #    model = RandomForestClassifier(n_estimators=50, random_state=42)
   #    model.fit(X_train, y_train)

   #    y_pred = model.predict(X_test)
   #    # rmse = mean_squared_error(y_test, y_pred)
   #    # mlflow.log_metric("rmse", rmse)

   #    mlflow.sklearn.log_model(model, "random_forest_classifier_model")
   #    print(f"Modelo com run Id: {run.info.run_id}")


def treinar_modelo(X, y, dataset_name, mlflow_dataset):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   print("após train teste split")
   param_grid = {
      "n_estimators": [50,100,200],
      "max_depth": [10,20,None],
      "min_samples_split": [2,5,10]
   }

   rf = RandomForestClassifier(random_state=42)
   print("após rf = RandomForestClassifier")


   for params in (dict (zip(param_grid.keys(), values)) for values in
                  [(n,d,s) for n in param_grid["n_estimators"]
                            for d in param_grid["max_depth"]
                            for s in param_grid["min_samples_split"]]):
      rf.set_params(**params)
      rf.fit(X_train, y_train)
      y_test_pred = rf.predict(X_test)
      accuracy =  accuracy_score(y_test, y_test_pred)
      precision = precision_score(y_test, y_test_pred, average='macro')
      recall = recall_score(y_test, y_test_pred, average='macro')

      with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}") as run:
         
         #dizer qual dataset usamos
         mlflow.log_input(mlflow_dataset, context="training")
         mlflow.log_params(params)
         mlflow.log_metric("accurary", accuracy)
         mlflow.log_metric("precision", precision)
         mlflow.log_metric("recall", recall)

         mlflow.set_tag("dataset_used", dataset_name)
         signature = infer_signature(X_train, y_test_pred)

         model_info = mlflow.sklearn.log_model(rf,"random_forest_model", 
                                                signature= signature, 
                                                input_example=X_train, 
                                                registered_model_name= "RandomForestClassifier_TFinal")
         
                                                # registered_model_name= f"RF_{params["n_estimators"]}_{params["max_depth"]}_{params["min_samples_split"]}")
         
         loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
         predictions = loaded_model.predict(X_test)
         result = pd.DataFrame(X_test, columns=X.columns.values)
         result["label"] = y_test.values
         result["predictions"] = predictions

         mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="classifier",
         )


         print("result[:5]")
         print(result[:5])
         print("run_finalizada ---- ")

         # mlflow.log_param("model_type", "RandomForestClassifier")
         
         # model = RandomForestClassifier(n_estimators=50, random_state=42)
         # model.fit(X_train, y_train)

         # y_pred = model.predict(X_test)
         # # rmse = mean_squared_error(y_test, y_pred)
         # # mlflow.log_metric("rmse", rmse)

         # mlflow.sklearn.log_model(model, "random_forest_classifier_model")
         # print(f"Modelo com run Id: {run.info.run_id}") 

   return rf

def getData(dataset_name):
#   dataframe = pd.read_csv(dataset_name)
  dataframe = pd.read_csv(dataset_name, sep=',')
  print("finalizou getData")

#   descreve_dataframe(dataframe)
  
  return dataframe

def descreve_dataframe(df):
   print("tamanho:")
   print(df.size)
   print("colunas:")
   print(df.columns)
   print("head:")
   print(df.head)

   primeira_linha = df.iloc[0]
   print(primeira_linha)



def preparacao_dados(df: pd.DataFrame):

   print("df.dtypes ANTESSSSSS 1")
   print(df.dtypes)

   df['date'] =  pd.to_datetime(df['date'])

   df['ano'] = df['date'].dt.year
   df['mes'] = df['date'].dt.month
   df['dia'] = df['date'].dt.day

   df = df.drop('date', axis=1)

   print("df.dtypes APOOOOOOSSS 22222")
   print(df.dtypes)

   # print("df['date']")
   # print(df['date'])
   print("AQUIIIIIIIIIIIII ----------------")
   # print(df['date'])
   # print(df.head)
   print("colunas")
   print(df.columns)

   for col in df.select_dtypes(include=["int32"]).columns:
      df[col] = df[col].astype("float64")

   for col in df.select_dtypes(include=["int64"]).columns:
      df[col] = df[col].astype("float64")

   for col in df.select_dtypes(include=["object"]).columns:
      df[col] = df[col].astype(str)
      
      #discretizacao
      df[col] = LabelEncoder().fit_transform(df[col])

   #para todos os dados ausentes, colocar valor 0
   # df.fi(0, inPlace=True)

   # print("antes do from_pandas")

   print("df.dtypes APOOOOOOSSS 33333333")
   print(df.dtypes)

   mlflow_dataset =  mlflow.data.pandas_dataset.from_pandas(df, targets="weather")
   # print("APOS do from_pandas")

   # print("df[date] ----")
   # print(df['date'])
   # df_1 = df[df['date'] < '2015-01-01']
   # df_2 = df[df['date'] >= '2015-01-01']

   df_1 = df[df['ano'] < 2015]
   df_2 = df[df['ano'] >= 2015]



   # print("df.dtypes ANTESSSSSS 333333")
   # print(df.dtypes)
   # print("df.dtypes APOOOOOOSSS 333333")
   # df_1['date'] = df_1['date'].astype(str)
   # print(df.dtypes)

   
   # print("df.dtypes ANTESSSSSS 444444")
   # print(df.dtypes)
   # print("df.dtypes APOOOOOOSSS 444444")

   # df_2['date'] = df_2['date'].astype(str)

   print("finalizou preparacao ----------------------------")
   # print("df.head")
   # print(df.head)
   # print("df_1.head")
   # print(df_1.head)
   # print("df_1.head")
   # print(df_1.head)
   return df_1,df_2,mlflow_dataset




if __name__ == "__main__":
    main()
 