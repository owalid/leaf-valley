FROM node:17.2-alpine

WORKDIR /home/app

COPY . /home/app/
ENV NODE_OPTIONS --openssl-legacy-provider
ENV NODE_ENV=dev
ENV BASE_API_URL="http://leaf-api:5000/api"
ENV NUXT_BASE_API_URL="http://leaf-api:5000/api"
ENV NUXT_HOST=0.0.0.0
ENV NUXT_PORT=3000

RUN yarn

CMD ["yarn", "dev"]

EXPOSE 3000
