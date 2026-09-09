import dagshub
import mlflow
mlflow.set_tracking_uri('https://dagshub.com/mayankkumar202001-svg/mylap_mlops_new.mlflow')
dagshub.init(repo_owner='mayankkumar202001-svg', repo_name='mylap_mlops_new', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)