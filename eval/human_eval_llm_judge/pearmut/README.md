```
pip install "pearmut==1.1.1"
pearmut add campaign.json
pearmut run --port 8002 --server https://pearmut.ngrok.dev
ngrok http --url=pearmut.ngrok.dev 8002
```