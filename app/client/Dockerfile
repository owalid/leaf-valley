FROM node:17.2-alpine

RUN mkdir -p /home/app
WORKDIR /home/app

ENV NODE_OPTIONS --openssl-legacy-provider
ENV NODE_ENV=dev
ENV BASE_API_URL="http://api:5000/api"
ENV NUXT_HOST=0.0.0.0
ENV NUXT_PORT=3000

RUN npm i

CMD ["npm", "run", "dev"]

EXPOSE 3000
