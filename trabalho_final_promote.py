import mlflow
from mlflow.tracking import MlflowClient
import subprocess

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
model_name = "RandomForestClassifier_TFinal"

# Definir os limites de F1-score para Staging e Production
staging_threshold = 0.56  # Apenas modelos acima deste F1-score vão para Staging

# Buscar todas as versões do modelo
#lista todas as versões do modelo. No caso, vão ser as 27 realizadas no codigo_trabalho_final_MLOPS.py
versions = client.search_model_versions(f"name='{model_name}'")

best_model = None  # Para armazenar o modelo Champion
best_f1_score = 0  # Para rastrear o melhor F1

# para cada versão, pegarei o run_Id para pegar as métricas. 
# Dentro dessas está o f1_score, que usaremos para estabelecer quem está acima do limite de .56 
# e também descobrir quais das versões criadas é a melhor para subir para production
# se f1_score da versão for maior que o 0.56 estabelecido acima, este será colocado como staging
# Após é feito o outro teste. Se a iteração atual for maior que a anterior, é trocado o melhor modelo

for version in versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics

    if "f1_score" in metrics:
        f1 = metrics["f1_score"]

        # Adicionar modelos qualificados para Staging
        if f1 > staging_threshold:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Staging"
            )
            print(f"Modelo versão {version.version} com F1-score {f1} movido para Staging.")

        # Encontrar o melhor modelo para Produção
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = version.version

# Atualizar o Champion (Produção)
# após terminar as iterações das versões, pego a versão que está no best_model e 
# coloco altero seu stage para Production
if best_model:
    client.transition_model_version_stage(
        name=model_name,
        version=best_model,
        stage="Production"
    )
    print(f"Modelo versão {best_model} agora é o Champion com F1-score {best_f1_score}.")
else:
    print("Nenhum modelo atende ao critério para ser Champion.")


print("registered_models ----------------")
registered_models = client.search_registered_models()

print("antes for registered_models ----------------")

for model in registered_models:
    for version in model.latest_versions:
        if version.current_stage == 'Production':
            print(f"Modelo: {version.name} - Versão: {version.version}")
            modelo_uri = f"models:/{version.name}/Production"
            break

print("Antes subprocess")
subprocess.Popen([
    "mlflow", "models", "serve",
    "-m", modelo_uri,
    "-p", "8000",
    "--env-manager", "virtualenv",
    "--no-conda"
])

print("APOS subprocess")


