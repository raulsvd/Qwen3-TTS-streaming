# Session: Disable Anonymous Auth for AgentIQ Backend

**Date**: 2025-01-15  
**Branch**: `feat/rag-polish-pack`  
**Status**: Complete - Ready for commit

## What Was Done

Disabled anonymous access for the agentic-ai backend now that agent-websites forwards auth headers properly.

### Files Changed

1. **`apps/agentic-ai/backend/src/config.py`**
   - Added `authentik_allow_anonymous: bool = False`
   - Env var: `AUTHENTIK_ALLOW_ANONYMOUS`

2. **`apps/agentic-ai/backend/src/middleware/auth.py`**
   - Updated dispatch() to check `authentik_allow_anonymous` flag
   - Returns 401 when no token and anonymous disabled
   - Returns 500 when auth required but issuer not configured

3. **`apps/agentic-ai/backend/tests/test_api.py`**
   - Added `TestAuthEnforcement` class with 3 tests

### Behavior Matrix

| Token | allow_anonymous | authentik_issuer | Result |
|-------|-----------------|------------------|--------|
| None  | False           | Set              | 401    |
| None  | False           | Empty            | 500    |
| None  | True            | Any              | Pass   |
| Valid | Any             | Set              | Pass   |

### Verification Commands

```bash
# Run tests
cd apps/agentic-ai/backend
python -m pytest tests/test_api.py::TestAuthEnforcement -v

# All tests pass (14 total in test_api.py)
```

### Production Verification

After deploy, test with:
```bash
# Should return 401
curl -X POST https://agentiq-backend-xxx.run.app/api/copilotkit \
  -H "Content-Type: application/json" \
  -d '{"messages":[]}'

# Should work with valid JWT from agent-websites
curl -X POST https://agentiq-backend-xxx.run.app/api/copilotkit \
  -H "Authorization: Bearer <jwt-from-agent-websites>" \
  -H "Content-Type: application/json" \
  -d '{"messages":[]}'
```

### No Terraform Changes Needed

Default `False` in code means auth is required without any env var changes.

## Related Context

- Previous work: Agent-websites now forwards `Authorization` header to backend
- The `authentik_issuer` and `authentik_client_id` are already configured via Doppler
- Health endpoints (`/health`, `/ready`) and internal governance endpoints bypass auth
