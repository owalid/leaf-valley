FROM node:14-alpine


WORKDIR /home/app

COPY docker-entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

ENV NODE_ENV=dev
ENV NUXT_PORT=3000


CMD ["yarn", "dev"]

EXPOSE 3000
