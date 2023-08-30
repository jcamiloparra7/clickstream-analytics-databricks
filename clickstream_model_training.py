# Databricks notebook source
import os

import mlflow
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef
from xgboost import XGBClassifier

# COMMAND ----------

os.environ['MODEL_NAME'] = 'clickstream_predictor'
os.environ['DATA_LAKE_PATH'] = 'dbfs:/FileStore/juanc_parra/clickstream'

# COMMAND ----------

MODEL_NAME = os.environ.get('MODEL_NAME')
DATA_LAKE_PATH = os.environ.get('MODEL_NAME')

# COMMAND ----------

# Reading the table "features_and_labels" into a DataFrame
features_and_labels = spark.read.table("features_and_labels")

# Reading the table "session_product_outcomes" into a DataFrame
session_product_outcomes = spark.read.table("session_product_outcomes")

# Selecting distinct 'user_id' and 'user_session' pairs from "features_and_labels" DataFrame
sessions = features_and_labels.select('user_id', 'user_session').distinct()

# Defining holdout ratio for data segmentation 
holdout_ratio = 0.1
train_test_ratio = 1 - holdout_ratio

# Splitting original data into "train_test" (90%) and "validate" (10%)
train_test, validate = sessions.randomSplit([train_test_ratio, holdout_ratio])

# Further splitting "train_test" into "train" and "test" subsets
train, test = train_test.randomSplit([1-(holdout_ratio/train_test_ratio), holdout_ratio/train_test_ratio])

# Joining corresponding 'user_id' and 'user_session' from "features_and_labels" to "train_test"
train_test = train_test.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')

# Joining corresponding 'user_id' and 'user_session' from "features_and_labels" to "train"
train = train.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')

# Joining corresponding 'user_id' and 'user_session' from "features_and_labels" to "test"
test = test.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')

# Joining corresponding 'user_id' and 'user_session' from "features_and_labels" to "validate"
validate = validate.join(features_and_labels, on=['user_id','user_session']).drop('user_id','user_session')

# COMMAND ----------

# Broadcast training data to all executor nodes
train_pd_broadcast = sc.broadcast(train.toPandas())

# Broadcast testing data to all executor nodes
test_pd_broadcast = sc.broadcast(test.toPandas())


# Define the search space for hyperparameters
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 20, 1),                       
    'learning_rate' : hp.uniform('learning_rate', 0.01, 0.40),
    'scale_pos_weight' : hp.uniform('scale_pos_weight', 1.0, 100)   
}

# Set up MLflow experiment 
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f'/Users/{username}/clickstream')

# COMMAND ----------

def evaluate_model(hyperopt_params):
  
  # Acquire replicated input data
  train_input = train_pd_broadcast.value
  test_input = test_pd_broadcast.value
  
  # Prepare training and testing data
  X_train = train_input.drop('purchased', axis=1)
  y_train = train_input['purchased']
  X_test = test_input.drop('purchased', axis=1)
  y_test = test_input['purchased']
  
  # Set up model parameters
  params = hyperopt_params
  # Convert these hyperparameters to integer as hyperopt provides them as float by default
  if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
  if 'min_child_weight' in params: params['min_child_weight'] = int(params['min_child_weight']) 
  if 'max_delta_step' in params: params['max_delta_step'] = int(params['max_delta_step']) 
  
  # Initialize model with chosen parameters
  model = XGBClassifier(**params)
  
  # Fit the model on training data
  model.fit(X_train, y_train)
  
  # Make predictions for testing data
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)
  
  # Compute evaluation metrics
  model_ap = average_precision_score(y_test, y_prob[:,1])
  model_ba = balanced_accuracy_score(y_test, y_pred)
  model_mc = matthews_corrcoef(y_test, y_pred)
  
  # Log metrics with MLFlow run
  mlflow.log_metrics({
    'avg precision':model_ap,
    'balanced_accuracy':model_ba,
    'matthews_corrcoef':model_mc
    })                                       
                                             
  # Invert key metric for hyperopt optimization
  loss = -1 * model_ap
  
  # Return results dictionary
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------


 
# perform evaluation
with mlflow.start_run(run_name='tuning'):
  
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=25,
    trials=SparkTrials(parallelism=5),
    verbose=True
    )

# separate hyperopt output from our results
print('\n')
 


# COMMAND ----------

# Start a new Mlflow run and perform evaluation
with mlflow.start_run(run_name='tuning'):

  # Call function to find the minimum
  argmin = fmin(
    fn=evaluate_model,          # The function to minimize
    space=search_space,         # Our hyperparameters search space
    algo=tpe.suggest,           # Algorithm for searching the space
    max_evals=25,               # Maximum number of evaluations on the objective function
    trials=SparkTrials(parallelism=5), # SparkTrials class to parallelize the hyperparameter tuning across Spark workers
    verbose=True                # Verbose flag
    )

# Print an empty line
print('\n')

# COMMAND ----------

# Retrieve optimized hyperparameters from the search space
hyperopt_params = space_eval(search_space, argmin)

# COMMAND ----------

with mlflow.start_run(run_name='trained'):
    # Convert Spark DataFrame to pandas DataFrame for the training and validation sets
    train_test_pd = train_test.toPandas()
    validate_pd = validate.toPandas()

    # Separate features (X) and target variable (y) for both training and validation sets
    X_train_test = train_test_pd.drop('purchased', axis=1)
    y_train_test = train_test_pd['purchased']
    X_validate = validate_pd.drop('purchased', axis=1)
    y_validate = validate_pd['purchased']

    # Configure model parameters
    params = hyperopt_params
    # Routine to convert float values to int as Hyperopt presents numeric parameter suggestions as floats
    if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params: params['min_child_weight'] = int(params['min_child_weight'])
    if 'max_delta_step' in params: params['max_delta_step'] = int(params['max_delta_step'])

    # Train the XGBoost classifier with the given parameters
    model = XGBClassifier(**params)
    model.fit(X_train_test, y_train_test)

    # Generate class predictions and class probabilities for the training set
    y_pred = model.predict(X_train_test)
    y_prob = model.predict_proba(X_train_test)

    # Calculate evaluation metrics
    model_ap = average_precision_score(y_train_test, y_prob[:,1])
    model_ba = balanced_accuracy_score(y_train_test, y_pred)
    model_mc = matthews_corrcoef(y_train_test, y_pred)

    # Log metrics with MLflow for tracking and comparison
    mlflow.log_metrics({
        'avg precision': model_ap,
        'balanced_accuracy': model_ba,
        'matthews corrcoef': model_mc
    })

    # Log the trained model with MLflow for reproducibility and deployment purposes  
    mlflow.sklearn.log_model(
        artifact_path='model',
        sk_model=model,
        pip_requirements=['xgboost'],
        registered_model_name=MODEL_NAME,
        pyfunc_predict_fn='predict_proba'
    )

    print(model_ap)

# COMMAND ----------

# Establish a connection to MLflow client
client = mlflow.tracking.MlflowClient()

client.search_model_versions(f"name = '{MODEL_NAME}'")[0].version

# COMMAND ----------

# Establish a connection to MLflow client
client = mlflow.tracking.MlflowClient()

# Fetch the most recent version of the model from the registry using the defined model name
model_version = client.search_model_versions(f"name = '{MODEL_NAME}'")[0].version

# Transition the identified model version to 'Production' stage 
# This action signifies that the model is now ready for use in production environment
try:
    client.transition_model_version_stage(
    name=MODEL_NAME,
    version=model_version,
    stage='production'
    )
except Exception:
    print('Model already in production stage')

# Create a User Defined Function (UDF) that wraps the model loaded from the specified path.
# This UDF is compatible with PySpark, courtesy of mlflow.sklearn.pyfunc.spark_udf method.
# The UDF can be used to execute predictions in Spark DataFrames.
predict_udf = mlflow.sklearn.pyfunc.spark_udf(
  spark,
  client.search_model_versions(f"name = '{MODEL_NAME}'")[0].source
)
