```
podman build -t rag_production .

podman run --rm \
  -p 8000:8000 \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production

  ```