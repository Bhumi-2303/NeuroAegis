FROM node:20-alpine as build

WORKDIR /app

# Copy root package.json for workspaces
COPY package.json package-lock.json ./
COPY packages/ ./packages/
COPY apps/web/package.json ./apps/web/

# Install dependencies
RUN npm ci

# Copy the rest of the web app
COPY apps/web/ ./apps/web/

# Build
WORKDIR /app/apps/web
RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=build /app/apps/web/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
