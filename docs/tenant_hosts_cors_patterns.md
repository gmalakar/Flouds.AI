# Trusted Hosts & CORS Patterns

The system supports flexible pattern formats for `trusted_hosts` and `cors_origins`.
Operators may configure these lists per-tenant or globally.

Supported formats:

- Exact match: `example.com` or `https://app.example.com`
- Wildcard: `*.example.com` — matches `example.com` and any subdomain like `api.example.com`.
- Full wildcard: `*` — allow all hosts/origins.
- Regex: prefix with `re:` and provide a full regular expression (for advanced cases). Example: `re:^(.+\.)?example\\.org$`.

Notes:

- A pattern beginning with `*.` will match both the root domain and subdomains (e.g. `*.example.com` matches `example.com` and `api.example.com`).
- Regex entries are evaluated with full-match semantics; ensure expressions are carefully reviewed and only allowed for trusted administrators.
- For CORS, checks are applied against both the full origin string and the hostname parsed from the origin.

Example app startup configuration:

```py
app.state.trusted_hosts = ["example.com", "*.example.com", "re:^allowed-\\d+\\.local$"]
app.state.cors_origins = ["https://app.example.com", "*.example.com"]
```
