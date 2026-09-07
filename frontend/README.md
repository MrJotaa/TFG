# Frontend

Frontend de la plataforma desarrollado con React y Vite.

La puesta en marcha completa de la aplicación, incluyendo frontend, backend y MongoDB, se encuentra documentada en el [README principal](../README.md).

## Desarrollo local

Instalar las dependencias:

```bash
npm ci
```

Iniciar el servidor de desarrollo:

```bash
npm run dev
```

Ejecutar las pruebas:

```bash
npm test
```

Ejecutar el análisis estático:

```bash
npm run lint
```

Generar el build de producción:

```bash
npm run build
```

Para la ejecución normal de la aplicación se recomienda utilizar Docker Compose desde la raíz del proyecto:

```bash
docker compose up --build
```
