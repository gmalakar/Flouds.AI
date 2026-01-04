# Tenant Config Roles and Semantics

Roles:

- `superadmin`: can manage global config (no tenant_id) and any tenant-scoped config.
- `admin`: can manage tenant-scoped config but must specify `tenant_id` explicitly.
- `tenant-admin`: can manage config only for their own tenant (identified via `X-Tenant-Code` or token claim).

Endpoints (example):

- POST `/api/v1/admin/config/set` — body: `key`, `value`, optional `tenant_id`.
- GET `/api/v1/admin/config/get` — query: `key`, optional `tenant_id`.
- DELETE `/api/v1/admin/config/delete` — query: `key`, optional `tenant_id`.

Security note: replace header-based demo role checks with validated tokens in production.
