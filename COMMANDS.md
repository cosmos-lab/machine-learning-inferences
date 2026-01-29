```
../../tools/llmcode/export_code.sh \
  ./app,./mlops,./models \
  ./Dockerfile,Dockerfile.mlops,requirements.txt \
  __pycache__,artifacts \
  output.txt


../../tools/llmcode/import_code.sh input.txt ./


podman build -f Dockerfile -t rag_production_standard:latest .


podman run --rm -p 8000:8000 \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production_standard:latest


```