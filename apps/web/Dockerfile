# Stage 1: Build the React application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy root workspace configurations and package definitions
COPY package.json package-lock.json ./
COPY packages/model-contracts/package.json ./packages/model-contracts/
COPY apps/web/package.json ./apps/web/

# Install workspace dependencies
RUN npm ci

# Copy source trees for local packages and web application
COPY packages/model-contracts ./packages/model-contracts
COPY apps/web ./apps/web

# Build arguments for API endpoints baked into static bundle
ARG VITE_API_BASE_URL=http://localhost:8000/api/v1
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_API_URL=$VITE_API_URL

# Build web application bundle
RUN npm run --workspace=@neuroaegis/web build

# Stage 2: Production Nginx server
FROM nginx:alpine

# Copy compiled frontend assets
COPY --from=builder /app/apps/web/dist /usr/share/nginx/html

# Copy SPA routing Nginx config
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
