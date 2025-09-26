import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def run_ragas_evaluation(json_path: str, output_prefix: str = "ragas_results"):
    """
    Ejecuta evaluación con RAGAS a partir de un archivo JSON.
    
    Args:
        json_path (str): Ruta al archivo .json con test_data
        output_prefix (str): Prefijo para los archivos de salida (.json y .csv)
    
    Returns:
        pd.DataFrame: DataFrame con los resultados de la evaluación
    """
    # 1. Cargar API keys desde .env
    load_dotenv()

    logging.info(f"Leyendo datos desde {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logging.info(f"Se cargaron {len(test_data)} ejemplos de test.")

    # 2. Convertir a Dataset de Hugging Face
    eval_dataset = Dataset.from_dict({
        "question": [item["question"] for item in test_data],
        "answer": [item["answer"] for item in test_data],
        "contexts": [item["contexts"] for item in test_data],
        "ground_truth": [item["ground_truth"] for item in test_data],
    })
    logging.info("Dataset preparado para evaluación.")

    # 3. Definir métricas
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    logging.info(f"Métricas seleccionadas: {[m.__class__.__name__ for m in metrics]}")
    
    # 4. Evaluar cada ejemplo individualmente
    results_list = []
    for idx, example in enumerate(test_data, start=1):
        try:
            dataset = Dataset.from_list([example])
            result = evaluate(dataset, metrics=metrics)
            df = result.to_pandas()
            results_list.append(df.iloc[0].to_dict())
            logging.info(f"Ejemplo {idx}/{len(test_data)} evaluado correctamente.")
        except Exception as e:
            logging.error(f"Error evaluando ejemplo {idx}: {e}")
            # Guardamos el error para poder rastrear luego
            results_list.append({"error": str(e), **example})

    # 5. Guardar resultados
    df_results = pd.DataFrame(results_list)
    csv_path = f"{output_prefix}.csv"
    json_path_out = f"{output_prefix}.json"
    df_results.to_csv(csv_path, index=False)
    df_results.to_json(json_path_out, orient="records", indent=2, force_ascii=False)

    logging.info(f"Resultados guardados en {csv_path} y {json_path_out}")

    return df_results


# Ejemplo de uso:
if __name__ == "__main__":
    df_results = run_ragas_evaluation("general_test.json", output_prefix="ragas_general_test")
    print(df_results)
