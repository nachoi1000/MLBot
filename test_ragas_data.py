import json
import logging
from langchain_core.messages import HumanMessage
import uuid


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from agent import agent_app


general_test = [
  {
    "question": "What is the primary motivation for supervised learning?",
    "ground_truth": "The primary motivation for supervised learning is to build a statistical model for predicting or estimating an output based on one or more inputs. This is useful in scenarios where we have a set of features (predictors) and a known outcome (response) and we want to predict the outcome for new, unseen data."
  },
  {
    "question": "What are the two main types of problems that supervised learning addresses?",
    "ground_truth": "Supervised learning addresses two main types of problems: regression and classification. Regression problems involve predicting a continuous or quantitative output, while classification problems involve predicting a qualitative or categorical output."
  },
  {
    "question": "Describe the difference between training data and test data in the context of machine learning.",
    "ground_truth": "Training data is the set of observations used to teach or train a statistical learning model. Test data, on the other hand, consists of previously unseen observations that are used to evaluate the performance and accuracy of the trained model."
  },
  {
    "question": "¿Cuál es la diferencia entre un problema de regresión y uno de clasificación?",
    "ground_truth": "Un problema de regresión se refiere a la predicción de un valor de salida cuantitativo o continuo, como el precio de una acción. Un problema de clasificación se refiere a la predicción de un valor de salida cualitativo o categórico, como si un correo electrónico es spam o no."
  },
  {
    "question": "What is the primary purpose of cross-validation?",
    "ground_truth": "Cross-validation is a technique used to estimate the test error of a statistical learning method, which helps in evaluating its performance and selecting the appropriate level of model complexity."
  },
  {
    "question": "Explain the difference between the validation set approach and leave-one-out cross-validation (LOOCV).",
    "ground_truth": "In the validation set approach, the data is randomly split into a training set and a validation set. The model is trained on the training set and evaluated on the validation set. In LOOCV, each observation is used as a validation set once, and the model is trained on the remaining n-1 observations. This process is repeated n times, and the test error is estimated by averaging the n resulting MSEs."
  },
  {
    "question": "What is the 'bias-variance trade-off' in the context of k-fold cross-validation?",
    "ground_truth": "In k-fold cross-validation, there is a trade-off between bias and variance. A smaller value of k (like 5 or 10) leads to a more biased estimate of the test error because the training sets are smaller than the full dataset. A larger value of k (like in LOOCV) leads to a less biased estimate but higher variance because the training sets are highly correlated. Typically, k=5 or k=10 is used as a compromise between bias and variance."
  },
  {
    "question": "¿Qué es el 'bootstrap' y para qué se utiliza en el aprendizaje estadístico?",
    "ground_truth": "El bootstrap es una herramienta estadística que consiste en tomar muestras repetidas con reemplazo de un conjunto de datos para obtener información adicional sobre un modelo ajustado. Se utiliza comúnmente para estimar la variabilidad de un estimador de parámetros o de un método de aprendizaje estadístico."
  },
  {
    "question": "What are two common metrics used to assess the accuracy of a linear regression model?",
    "ground_truth": "Two common metrics are the Residual Standard Error (RSE), which represents the average deviation of the response from the true regression line, and the R-squared (R²) statistic, which measures the proportion of variability in the response that can be explained by the predictors."
  },
  {
    "question": "What is the interpretation of an interaction term in a multiple linear regression model?",
    "ground_truth": "An interaction term, such as Xj * Xk, in a multiple linear regression model suggests that the effect of one predictor on the response variable depends on the level of another predictor. This is also known as a synergy effect."
  },
  {
    "question": "How can you detect collinearity in a multiple regression model, and what are two ways to address it?",
    "ground_truth": "Collinearity can be detected by examining the correlation matrix of the predictors or by calculating the Variance Inflation Factor (VIF). Two ways to address it are to either drop one of the problematic variables or to combine the collinear variables into a single predictor."
  },
  {
    "question": "¿Qué es una variable ficticia ('dummy variable') y cuándo se utiliza en la regresión lineal?",
    "ground_truth": "Una variable ficticia (dummy variable) es una variable que toma valores de 0 o 1 para indicar la ausencia o presencia de alguna categoría o atributo. Se utiliza para incorporar predictores cualitativos en un modelo de regresión lineal, permitiendo analizar el efecto de diferentes categorías en la variable de respuesta."
  },
  {
    "question": "What is the primary difference between Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)?",
    "ground_truth": "The primary difference between LDA and QDA is the assumption about the covariance matrix of the predictors. LDA assumes that all classes share a common covariance matrix, resulting in a linear decision boundary. QDA, on the other hand, allows each class to have its own covariance matrix, leading to a quadratic decision boundary."
  },
  {
    "question": "What is the 'naive' assumption made by the Naive Bayes classifier?",
    "ground_truth": "The Naive Bayes classifier makes the 'naive' assumption that within each class, the predictor variables are independent of each other. This simplification allows for easier estimation of the joint probability distribution."
  },
  {
    "question": "How does logistic regression handle binary classification problems?",
    "ground_truth": "Logistic regression models the probability that the response variable belongs to a particular category by using the logistic function. This function takes a linear combination of the predictors and transforms it into a value between 0 and 1, which represents the probability of the outcome."
  },
  {
    "question": "¿Qué es la curva ROC y qué representa el área bajo la curva (AUC)?",
    "ground_truth": "La curva ROC (Receiver Operating Characteristic) es una gráfica que muestra el rendimiento de un modelo de clasificación binaria para todos los posibles umbrales de clasificación. El área bajo la curva (AUC) representa una medida agregada del rendimiento del clasificador; un valor de AUC cercano a 1 indica un modelo muy bueno, mientras que un valor de 0.5 indica un rendimiento similar al azar."
  },
  {
    "question": "What is the difference between a forward stepwise selection and a backward stepwise selection method?",
    "ground_truth": "Forward stepwise selection starts with no predictors and adds them one at a time, based on which addition provides the best improvement to the model. Backward stepwise selection starts with all predictors and removes them one by one, eliminating the least significant predictor at each step."
  },
  {
    "question": "Explain the concept of 'shrinkage' or 'regularization' in the context of linear models like ridge regression and lasso.",
    "ground_truth": "Shrinkage or regularization is a technique used to constrain or reduce the coefficient estimates of a linear model towards zero. This helps to reduce the model's variance and prevent overfitting, particularly when dealing with a large number of predictors."
  },
  {
    "question": "What is the key difference between ridge regression and the lasso?",
    "ground_truth": "The key difference is that ridge regression uses an L2 penalty, which shrinks the coefficients towards zero but never sets them exactly to zero. The lasso, on the other hand, uses an L1 penalty, which can force some coefficient estimates to be exactly zero, effectively performing variable selection."
  },
  {
    "question": "¿Qué es el 'Principio de jerarquía' en la selección de modelos?",
    "ground_truth": "El principio de jerarquía establece que si se incluye un término de interacción en un modelo, también se deben incluir los efectos principales correspondientes, incluso si sus coeficientes no son estadísticamente significativos. Esto se debe a que la interacción suele estar correlacionada con los efectos principales, y omitirlos puede cambiar el significado de la interacción."
  },
  {
    "question": "What are regression splines and how do they differ from polynomial regression?",
    "ground_truth": "Regression splines are a method for fitting non-linear relationships by dividing the range of the predictor variable into distinct regions and fitting a separate polynomial in each region. Unlike polynomial regression, which fits a single high-degree polynomial over the entire range, splines are more flexible and stable, especially at the boundaries, because they use lower-degree polynomials in each segment."
  },
  {
    "question": "What is the purpose of a smoothing spline?",
    "ground_truth": "A smoothing spline aims to find a function that fits the data well by minimizing the residual sum of squares, while simultaneously penalizing the roughness of the function. This is achieved by adding a penalty term based on the integral of the squared second derivative of the function, controlled by a tuning parameter lambda."
  },
  {
    "question": "How does local regression work?",
    "ground_truth": "Local regression, or LOESS, is a non-parametric method that fits a separate regression model for each target point, using only the data points in its local neighborhood. It assigns weights to these neighboring points based on their distance from the target point, and then fits a weighted least squares regression. This results in a smooth and flexible curve that adapts to the local structure of the data."
  },
  {
    "question": "¿Qué es un modelo aditivo generalizado (GAM) y cuál es su principal ventaja?",
    "ground_truth": "Un modelo aditivo generalizado (GAM) es una extensión de los modelos lineales que permite que la relación entre cada variable predictora y la variable de respuesta sea no lineal, mientras se mantiene la aditividad de los efectos. Su principal ventaja es la capacidad de modelar relaciones complejas y no lineales de manera flexible, al mismo tiempo que se puede examinar el efecto de cada predictor individualmente mientras se mantienen los demás constantes."
  },
  {
    "question": "What is the key difference between bagging and boosting?",
    "ground_truth": "Bagging involves creating multiple independent bootstrap samples of the training data and fitting a separate model to each, with the final prediction being the average of all models. Boosting, in contrast, builds models sequentially, where each new model is fitted to the residuals of the previous ones, thus focusing on the errors of the prior models."
  },
  {
    "question": "How does a random forest improve upon a bagged tree model?",
    "ground_truth": "A random forest improves on bagging by introducing an additional layer of randomness. When building each tree in the forest, at each split, only a random subset of predictors is considered as split candidates. This decorrelates the trees, reducing the variance of the overall model and often leading to better predictive accuracy."
  },
  {
    "question": "What is the role of the shrinkage parameter in boosting?",
    "ground_truth": "The shrinkage parameter, often denoted by lambda, in a boosting algorithm controls the learning rate. A smaller value for lambda slows down the learning process, requiring more trees to be added to the model but generally leading to improved performance by reducing the risk of overfitting."
  },
  {
    "question": "¿Qué es el sobreajuste (overfitting) en el contexto de los árboles de decisión y cómo se puede evitar?",
    "ground_truth": "El sobreajuste en los árboles de decisión ocurre cuando el árbol es demasiado complejo y se ajusta a los datos de entrenamiento tan bien que captura el ruido en lugar de la señal subyacente. Esto resulta en un bajo rendimiento en datos nuevos. Para evitarlo, se puede podar el árbol, es decir, reducir su tamaño, o utilizar métodos de conjunto como bagging, random forests o boosting, que combinan múltiples árboles para mejorar la generalización."
  },
  {
    "question": "What is the role of a kernel in a Support Vector Machine (SVM)?",
    "ground_truth": "A kernel in an SVM is a function that allows the algorithm to implicitly map the input data into a higher-dimensional feature space. This enables the SVM to find a non-linear decision boundary in the original feature space by constructing a linear separating hyperplane in the transformed space. Common kernels include polynomial and radial basis function (RBF) kernels."
  },
  {
    "question": "What is the difference between a maximal margin classifier and a support vector classifier?",
    "ground_truth": "A maximal margin classifier requires that the data be perfectly separable by a hyperplane, and it finds the hyperplane that maximizes the margin (the distance to the nearest training points). A support vector classifier is an extension that allows for some misclassifications by introducing a 'soft margin', which makes it applicable to data that is not linearly separable."
  },
  {
    "question": "What are support vectors in the context of SVMs?",
    "ground_truth": "Support vectors are the training data points that lie on or within the margin of the separating hyperplane. These are the critical points that define the hyperplane, and if they were to be moved, the hyperplane would also move."
  },
  {
    "question": "¿Cuáles son las dos estrategias más comunes para extender SVMs a problemas de clasificación multiclase?",
    "ground_truth": "Las dos estrategias más comunes para extender las SVMs a problemas de clasificación multiclase son 'uno contra uno' (one-vs-one) y 'uno contra todos' (one-vs-all). El enfoque de uno contra uno construye un clasificador para cada par de clases, y el de uno contra todos construye un clasificador por cada clase, que la distingue de todas las demás clases combinadas."
  },
  {
    "question": "What are the advantages of a deep learning model compared to a simple neural network?",
    "ground_truth": "Deep learning models, which are neural networks with multiple hidden layers, can learn more complex, hierarchical representations of the data. Each layer learns to extract features from the output of the previous layer, allowing the model to learn intricate patterns that might be missed by a shallow network. This often leads to superior performance on complex tasks like image and speech recognition."
  },
  {
    "question": "What is a convolutional neural network (CNN) and what kind of data is it particularly well-suited for?",
    "ground_truth": "A convolutional neural network (CNN) is a type of deep learning model that uses convolutional and pooling layers to automatically and adaptively learn spatial hierarchies of features from data. They are particularly well-suited for processing grid-like data such as images and videos, where local patterns are important."
  },
  {
    "question": "Explain the concept of 'dropout' in the context of training neural networks.",
    "ground_truth": "Dropout is a regularization technique used in training neural networks to prevent overfitting. During training, a certain proportion of neurons (and their connections) are randomly ignored or 'dropped out' for each training sample. This forces the network to learn more robust features and prevents it from becoming overly reliant on any single neuron."
  },
  {
    "question": "¿Qué es una red neuronal recurrente (RNN) y para qué tipo de datos es adecuada?",
    "ground_truth": "Una red neuronal recurrente (RNN) es un tipo de red neuronal diseñada para procesar datos secuenciales, como texto, series de tiempo o habla. A diferencia de las redes neuronales de avance, las RNN tienen conexiones recurrentes que les permiten mantener una memoria interna de la información procesada anteriormente, lo que las hace adecuadas para tareas en las que el contexto y el orden de los datos son importantes."
  },
  {
    "question": "What is the difference between Principal Component Analysis (PCA) and clustering?",
    "ground_truth": "PCA is a dimensionality reduction technique that aims to find a low-dimensional representation of the data that captures the maximum amount of variance. Clustering, on the other hand, is an unsupervised learning method that aims to group similar data points together into clusters based on their features."
  },
  {
    "question": "What is the main advantage of hierarchical clustering over K-means clustering?",
    "ground_truth": "The main advantage of hierarchical clustering is that it does not require the number of clusters (K) to be specified in advance. It produces a dendrogram, which is a tree-like diagram that allows for the visualization of the data's hierarchical structure and helps in choosing the number of clusters after the fact."
  },
  {
    "question": "Explain the concept of a 'scree plot' and how it is used in PCA.",
    "ground_truth": "A scree plot is a graphical representation of the eigenvalues or the proportion of variance explained (PVE) by each principal component. It is used to determine the optimal number of principal components to retain by identifying the 'elbow' in the plot, which is the point where the PVE drops off significantly."
  },
  {
    "question": "¿Qué significa que los datos están 'censurados' en el análisis de supervivencia?",
    "ground_truth": "En el análisis de supervivencia, los datos están censurados cuando el evento de interés no se ha observado para un individuo durante el período de estudio. Esto puede suceder si el estudio finaliza antes de que ocurra el evento para ese individuo o si se pierde el seguimiento de ese individuo. La información de que el evento no ha ocurrido hasta un cierto punto en el tiempo se utiliza en el análisis."
  },
  {
    "question": "What is the family-wise error rate (FWER) and why is it important in multiple hypothesis testing?",
    "ground_truth": "The family-wise error rate (FWER) is the probability of making at least one Type I error (falsely rejecting a true null hypothesis) among a set of hypothesis tests. It's important to control the FWER in multiple testing scenarios to avoid an inflated probability of making false discoveries, which can happen if you only control the individual Type I error rate for each test."
  },
  {
    "question": "How does the Benjamini-Hochberg procedure differ from the Bonferroni correction for multiple testing?",
    "ground_truth": "The Bonferroni correction controls the FWER by applying a very stringent significance level (alpha divided by the number of tests) to each individual test, making it conservative. The Benjamini-Hochberg procedure, on the other hand, controls the False Discovery Rate (FDR) and is generally more powerful. It orders the p-values and rejects hypotheses based on a dynamic threshold that depends on the rank of the p-value."
  },
  {
    "question": "Describe a scenario where a re-sampling approach, like permutation testing, would be preferred over a traditional parametric test.",
    "ground_truth": "A re-sampling approach, like permutation testing, is preferred when the assumptions of a parametric test are not met. For example, if the data is not normally distributed, especially with small sample sizes, the results of a t-test might not be valid. In such cases, permutation tests provide a more robust way to assess statistical significance without relying on distributional assumptions."
  },
  {
    "question": "¿Qué es la 'maldición de la dimensionalidad' y cómo afecta a los métodos de aprendizaje estadístico?",
    "ground_truth": "La 'maldición de la dimensionalidad' se refiere al hecho de que a medida que aumenta el número de características (dimensiones), los datos se vuelven cada vez más dispersos en el espacio de alta dimensión. Esto hace que los métodos de aprendizaje estadístico, especialmente los no paramétricos como k-vecinos más cercanos (KNN), sean menos efectivos, ya que la distancia entre los puntos de datos aumenta y la noción de 'vecindad' se vuelve menos significativa."
  }
]

formula_test = [
  {
    "question": "What is the primary motivation for supervised learning?",
    "ground_truth": "The primary motivation for supervised learning is to build a statistical model for predicting or estimating an output based on one or more inputs. This is useful in scenarios where we have a set of features (predictors) and a known outcome (response) and we want to predict the outcome for new, unseen data."
  },
  {
    "question": "What are the two main types of problems that supervised learning addresses?",
    "ground_truth": "Supervised learning addresses two main types of problems: regression and classification. Regression problems involve predicting a continuous or quantitative output, while classification problems involve predicting a qualitative or categorical output."
  },
  {
    "question": "Describe the difference between training data and test data in the context of machine learning.",
    "ground_truth": "Training data is the set of observations used to teach or train a statistical learning model. Test data, on the other hand, consists of previously unseen observations that are used to evaluate the performance and accuracy of the trained model."
  },
  {
    "question": "¿Cuál es la diferencia entre un problema de regresión y uno de clasificación?",
    "ground_truth": "Un problema de regresión se refiere a la predicción de un valor de salida cuantitativo o continuo, como el precio de una acción. Un problema de clasificación se refiere a la predicción de un valor de salida cualitativo o categórico, como si un correo electrónico es spam o no."
  },
  {
    "question": "According to the text, what is the formula for the t-statistic used in hypothesis testing for linear regression coefficients?",
    "ground_truth": "The t-statistic is calculated as t = (β̂₁ - 0) / SE(β̂₁), which measures how many standard deviations the coefficient estimate β̂₁ is away from 0."
  },
  {
    "question": "What is the mathematical definition of the R² statistic as provided in the book?",
    "ground_truth": "The R² statistic is defined as R² = (TSS - RSS) / TSS, which is equal to 1 - (RSS / TSS), where TSS is the total sum of squares and RSS is the residual sum of squares."
  },
  {
    "question": "What is the formula for the logit transformation in logistic regression?",
    "ground_truth": "The logit transformation is given by the formula log(p(X) / (1 - p(X))) = β₀ + β₁X₁ + ... + βₚXₚ. The left-hand side is referred to as the log-odds or logit."
  },
  {
    "question": "¿Cómo se calcula la estimación de la Máxima Verosimilitud (MLE) para un modelo de regresión logística?",
    "ground_truth": "Para estimar los coeficientes β₀ y β₁ en la regresión logística, se utiliza el método de máxima verosimilitud, que maximiza la función de verosimilitud ℓ(β₀, β₁) = Π_{i: y_i=1} p(x_i) * Π_{i': y_{i'}=0} (1 - p(x_{i'}))."
  },
  {
    "question": "In ridge regression, what is the shrinkage penalty term that is added to the Residual Sum of Squares (RSS)?",
    "ground_truth": "The shrinkage penalty in ridge regression is λ times the sum of the squares of the coefficients, represented as λΣ(β_j²)."
  },
  {
    "question": "How does the penalty term in LASSO regression differ from that in ridge regression?",
    "ground_truth": "The LASSO regression uses an L1 penalty, which is λ times the sum of the absolute values of the coefficients (λΣ|β_j|), whereas ridge regression uses an L2 penalty (λΣβ_j²)."
  },
  {
    "question": "What is the formula for the K-Nearest Neighbors (KNN) regression predictor?",
    "ground_truth": "The KNN regression predictor estimates f(x₀) by averaging the training responses in the neighborhood N₀, using the formula f̂(x₀) = (1/K) * Σ_{xᵢ ∈ N₀} yᵢ."
  },
  {
    "question": "¿Cuál es la fórmula para la regresión polinómica de grado d?",
    "ground_truth": "La fórmula para la regresión polinómica es yᵢ = β₀ + β₁xᵢ + β₂xᵢ² + ... + βₔxᵢᵈ + εᵢ."
  },
  {
    "question": "What is the objective function that a smoothing spline aims to minimize?",
    "ground_truth": "A smoothing spline minimizes the expression Σ(yᵢ - g(xᵢ))² + λ∫g''(t)²dt, where the first term is a loss function and the second is a penalty term that penalizes the variability in g."
  },
  {
    "question": "How is the effective degrees of freedom (dfλ) for a smoothing spline defined?",
    "ground_truth": "The effective degrees of freedom for a smoothing spline is defined as dfλ = Σ_{i=1 to n} {S_λ}_{ii}, where S_λ is the matrix that transforms the response vector y into the vector of fitted values ĝλ."
  },
  {
    "question": "What is the formula for a polynomial kernel of degree d in a Support Vector Machine?",
    "ground_truth": "A polynomial kernel of degree d is defined by the formula K(x_i, x_i') = (1 + Σ_{j=1 to p} x_ij * x_i'j)^d."
  },
  {
    "question": "¿Qué es la función de pérdida de bisagra (hinge loss) utilizada en los clasificadores de vectores de soporte?",
    "ground_truth": "La función de pérdida de bisagra se define como max[0, 1 - yᵢ(β₀ + β₁xᵢ₁ + ... + βₚxᵢₚ)]. Esta función es cero para las observaciones que están en el lado correcto del margen, lo que significa que solo los vectores de soporte afectan al clasificador."
  },
  {
    "question": "What are the two main steps involved in Principal Component Regression (PCR)?",
    "ground_truth": "The two steps are: first, constructing the first M principal components, Z₁, ..., Z_M, which are linear combinations of the original features. Second, fitting a linear regression model using these M components as predictors."
  },
  {
    "question": "What is the formula for the Proportion of Variance Explained (PVE) by the m-th principal component?",
    "ground_truth": "The PVE for the m-th principal component is given by the formula (Σ_{i=1 to n} z_{im}²) / (Σ_{j=1 to p} Σ_{i=1 to n} x_{ij}²), where z_{im} are the scores of the m-th principal component and x_{ij} are the centered data values."
  },
  {
    "question": "What is the optimization problem that defines K-means clustering?",
    "ground_truth": "K-means clustering aims to solve the problem: minimize Σ_{k=1 to K} (1/|C_k|) * Σ_{i, i' ∈ C_k} Σ_{j=1 to p} (x_ij - x_i'j)², which seeks to partition the observations into K clusters such that the total within-cluster variation is as small as possible."
  },
  {
    "question": "¿Cómo se define la distancia basada en la correlación entre dos observaciones?",
    "ground_truth": "La distancia basada en la correlación considera que dos observaciones son similares si sus características están altamente correlacionadas. La matriz de disimilitud se puede calcular como uno menos la matriz de correlación entre las observaciones."
  }
]


def get_rag_response(question: str):
    """
    Queries the LangGraph agent and extracts the answer and contexts.
    """
    # The 'messages' key in the agent's state is a list of all interactions.
    # We start a new conversation for each question in the test set.
    # The thread_id can be unique for each evaluation run or even each question.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # The input to the graph is a list of messages. We start with the user's question.
    inputs = {"messages": [HumanMessage(content=question)]}
    
    # Invoke the agent graph
    result_state = agent_app.invoke(inputs, config)
    
    # Extract the answer and context from the final state
    # The answer is the last message in the list, which is from the AI
    answer = result_state["messages"][-1].content
    
    # The retrieved documents are stored in 'retrieved_chunks'
    contexts = [doc.page_content for doc in result_state["retrieved_chunks"]]
    
    return {"answer": answer, "contexts": contexts}

def process_and_save_results(dataset: list, output_path: str):
    """
    Procesa una lista de preguntas, obtiene respuestas y contextos del agente RAG,
    y guarda los resultados en un archivo de texto (JSON).

    Args:
        dataset (list): Una lista de diccionarios, donde cada diccionario tiene una clave "question".
        output_path (str): La ruta del archivo donde se guardarán los resultados.
    """
    results_list = []
    
    for item in dataset:
        question = item.get("question")
        if not question:
            print(f"Advertencia: El item {item} no tiene una clave 'question'. Saltando.")
            continue
            
        # Obtiene la respuesta y los contextos del agente
        rag_output = get_rag_response(question)
        
        # Crea un nuevo diccionario combinando el original con la salida del RAG
        # Esto mantiene cualquier otra información que tuvieras (como 'ground_truth')
        updated_item = item.copy()
        updated_item['answer'] = rag_output['answer']
        updated_item['contexts'] = rag_output['contexts']
        results_list.append(updated_item)
        
    # Guarda la lista de resultados en un archivo de texto con formato JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        # Usamos json.dump con indent=4 para que el archivo sea legible
        json.dump(results_list, f, ensure_ascii=False, indent=4)
        
    print(f"\nResultados guardados exitosamente en: {output_path}")
    return results_list

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Llama a la función para procesar la lista y guardar los resultados
    processed_data_1 = process_and_save_results(formula_test, "formula_test.json")
    processed_data_2 = process_and_save_results(general_test, "formula_test.json")
    