mkdir -p /workspaces/ESG_Credit_Approval/mlflowlogs/mlruns
chmod -R 777 /workspaces/ESG_Credit_Approval/mlflowlogs
#!/bin/bash
mlflow ui --backend-store-uri file:///workspaces/ESG_Credit_Approval/mlflowlogs/mlruns --port 5000

