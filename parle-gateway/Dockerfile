# AI Gateway Server — Standalone Bun HTTP + WebSocket
#
# Build from project root: docker build -f backend/Dockerfile -t parle-gateway .

FROM oven/bun:1-alpine

WORKDIR /app

# Copy workspace structure
COPY package.json bun.lock tsconfig.json ./
COPY packages/ai-gateway/ ./packages/ai-gateway/

# Install: two-pass approach to handle lockfile mismatches
# Pass 1: try frozen lockfile (fast, cached)
# Pass 2: if frozen fails, do fresh install
RUN bun install --frozen-lockfile --ignore-scripts 2>/dev/null \
    || bun install --ignore-scripts --no-save 2>/dev/null \
    || (rm -rf node_modules packages/ai-gateway/node_modules && bun install --ignore-scripts 2>/dev/null; true)

# Ensure critical hoisted deps exist (workspace hoisting sometimes misses them)
RUN cd packages/ai-gateway && bun add franc zod 2>/dev/null; true

# Copy gateway code
COPY backend/gateway-server.ts ./backend/
COPY backend/ai-gateway-standalone.ts ./backend/

ENV SPEECH_GATEWAY_PORT=5050
EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD bun -e "const r = await fetch('http://localhost:5050/health'); if (!r.ok) process.exit(1);" || exit 1

CMD ["bun", "backend/gateway-server.ts"]
