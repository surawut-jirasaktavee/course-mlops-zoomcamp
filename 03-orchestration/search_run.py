from re import ASCII
import pandas as pd

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient  
    
class mlflow_client: 
    
    def __init__(self, mlflow_tracking_uri: str="sqlite:///mlflow.db", id: int=1):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.id = id
       
    def _client(self):        
        mlflow_tracking_uri = self.mlflow_tracking_uri 
        client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        return client
       
    def client(self): 
        return self._client()
    
    def _experiment(self):
        experiments = self.client().list_experiments()
        exp_ids = list(experiments[self.id].experiment_id)
        return exp_ids[-1]
   
    def experiment(self):
        return self._experiment()
    
    def _runs(self):
        
        runs = self.client().search_runs(
                                experiment_ids=self.experiment(),
                                filter_string="metrics.rmse < 6.5 and tags.model != 'hyperopt'",
                                run_view_type=ViewType.ACTIVE_ONLY,
                                max_results=10,
                                order_by=["metrics.rmse DESC"]                              
        )
        run_list = [run for run in runs]
        run = run_list[-1]
       
        return run
    
    def runs(self):
        return self._runs()
    
  