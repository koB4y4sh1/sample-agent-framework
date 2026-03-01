FROM node:20-alpine

RUN npm install -g supergateway

EXPOSE 8000

CMD ["supergateway", "--stdio", "stdin", "--outputTransport", "streamableHttp", "--port", "8000"]
