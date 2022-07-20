FROM node:17.2-alpine

WORKDIR /home/app

ENV NODE_OPTIONS --openssl-legacy-provider
ENV NODE_ENV=dev
ENV BASE_API_URL="http://api:5000/api"
ENV NUXT_HOST=0.0.0.0
ENV NUXT_PORT=3000

CMD ["yarn", "dev"]

EXPOSE 3000
