FROM node:17.2-alpine

RUN mkdir -p /home/app
COPY . /home/app/
WORKDIR /home/app

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

ENV NODE_OPTIONS --openssl-legacy-provider
ENV NODE_ENV dev
ENV BASE_API_URL "http://0.0.0.0:5000/api"
ENV NUXT_BASE_API_URL "http://0.0.0.0:5000/api"
ENV NUXT_HOST 0.0.0.0
ENV HOST 0.0.0.0
ENV NUXT_PORT 3000
ENV PORT 3000

CMD ["yarn", "dev"]

EXPOSE 3000
