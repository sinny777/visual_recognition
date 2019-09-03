# Running Client build in nginx Docker container

```
docker run --name ocr-client -p 8080:80 -v "$PWD":/usr/share/nginx/html:ro -it nginx
```