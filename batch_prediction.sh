PIPENV_PYTHON=/home/codespace/.local/share/virtualenvs/mlops-practice-2_cZx3tu/bin/python
ROOT_PATH=/workspace/mlops-practice

$PIPENV_PYTHON $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1
