FROM nginx:alpine
COPY assets /usr/share/nginx/html/assets
COPY index.html /usr/share/nginx/html/index.html

docker run --name ocr-client -p 8080:80 -v "$PWD":/usr/share/nginx/html:ro -it nginx