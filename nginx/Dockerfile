FROM nginx:alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY server.crt /etc/nginx/ssl/server.crt
COPY server.key /etc/nginx/ssl/server.key

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
