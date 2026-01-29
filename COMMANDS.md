```
./export_code.sh \
  ./app,./mlops,./models \
  ./Dockerfile,Dockerfile.mlops,requirements.txt \
  __pycache__,artifacts \
  output.txt


./import_code.sh output.txt ./test

```