FROM node:12-alpine 

RUN mkdir /home/node/app/ && chown -R node:node /home/node/app

# Create app directory
WORKDIR /home/node/app

# Install app dependencies
COPY --chown=node:node server/package*.json ./

# Change user to "node"
USER node

RUN npm install

# Bundle app source
COPY --chown=node:node server/ .

EXPOSE 8080

CMD [ "node", "server.js"]