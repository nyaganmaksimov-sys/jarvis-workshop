# Интеграция с A4Print-HUB

Jarvis Workshop должен оставаться отдельным сервисом. HUB получает только нормализованные данные через HTTP API.

## Базовые запросы

- `GET /health` — состояние Jarvis Workshop;
- `GET /api/v1/devices` — текущее состояние оборудования;
- `GET /api/v1/devices/{device_id}` — один станок;
- `GET /api/v1/events?limit=100` — последние события;
- `POST /api/v1/events` — входящие события от Edge Agent.

## Привязка производства

В следующей версии добавляется сущность assignment:

```json
{
  "device_id": "flashforge-ad5x-01",
  "production_job_id": "<hub-production-job-id>",
  "order_id": "<hub-order-id>"
}
```

Jarvis не должен самостоятельно менять заказ в HUB только по одному кадру. Изменения производственного статуса должны проходить через State Engine и отдельный HUB Adapter.

## План синхронизации

1. HUB создаёт/назначает производственное задание.
2. HUB Adapter связывает задание с `device_id`.
3. Edge Agent наблюдает оборудование.
4. State Engine подтверждает состояние.
5. Jarvis создаёт событие `job.started`, `job.completed` или `anomaly.detected`.
6. HUB Adapter обновляет производственное задание либо создаёт предупреждение.

## Безопасность

Для production использовать отдельный сервисный ключ между Edge Agent, Jarvis Server и HUB. Ключи не коммитить в GitHub.
