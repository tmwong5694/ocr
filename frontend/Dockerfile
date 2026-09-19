FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY index.html /usr/share/nginx/html/index.html
COPY app.js /usr/share/nginx/html/static/app.js
COPY styles.css /usr/share/nginx/html/static/styles.css