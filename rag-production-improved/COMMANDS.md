```
../../tools/llmcode/export_code.sh \
  ./app,./mlops \
  ./Dockerfile,Dockerfile.mlops,requirements.txt \
  __pycache__,artifacts \
  tmp.py \
  output.txt

../../tools/llmcode/import_code.sh output.txt ./

```