FROM nginx

RUN rm /etc/nginx/conf.d/default.conf

COPY site-config /site-config
RUN ln -s /site-config/site_nginx.conf /etc/nginx/conf.d/

CMD ["nginx", "-g", "daemon off;"]
