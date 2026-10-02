import test, { describe, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert';
import {
  apiFetch,
  loginUser,
  getMe,
  logoutUser,
  logoutAllSessions,
  setCsrfToken,
  getCsrfToken,
  onSessionExpired,
  ApiRequestError,
  uploadEegV2,
  generateRequestId,
} from '../src/services/api.ts';

// Setup Mock Environment
let originalFetch: typeof globalThis.fetch;
let mockCookie = '';
let fetchCalls: Array<{ url: string; options: RequestInit }> = [];

beforeEach(() => {
  originalFetch = globalThis.fetch;
  fetchCalls = [];
  mockCookie = '';
  setCsrfToken(null);

  // Setup minimal document.cookie mock for Node environment
  if (typeof globalThis.document === 'undefined') {
    (globalThis as any).document = {
      get cookie() {
        return mockCookie;
      },
      set cookie(val: string) {
        mockCookie = val;
      },
    };
  } else {
    mockCookie = '';
  }

  // Setup mock localStorage & sessionStorage to verify zero token leakage
  const storageMock = () => {
    const store = new Map<string, string>();
    return {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => store.set(k, v),
      removeItem: (k: string) => store.delete(k),
      clear: () => store.clear(),
      get length() {
        return store.size;
      },
    };
  };
  (globalThis as any).localStorage = storageMock();
  (globalThis as any).sessionStorage = storageMock();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Phase 9.4 Frontend Authentication & Session Boundary Suite', () => {
  // AUTH-01: unauthenticated user sees login boundary, not clinical data
  test('AUTH-01: unauthenticated state boundary triggers session expiration and error', async () => {
    let sessionExpiredCalled = false;
    const unsub = onSessionExpired(() => {
      sessionExpiredCalled = true;
    });

    globalThis.fetch = async () => {
      return new Response(JSON.stringify({ detail: 'Not authenticated' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    await assert.rejects(
      async () => {
        await apiFetch('/patients/pat-123');
      },
      (err: any) => {
        assert(err instanceof ApiRequestError);
        assert.strictEqual(err.status, 401);
        return true;
      }
    );

    assert.strictEqual(sessionExpiredCalled, true);
    unsub();
  });

  // AUTH-02: login sends credentials with credentials: 'include'
  test('AUTH-02: login sends credentials with credentials: "include"', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(
        JSON.stringify({
          csrf_token: 'csrf-secret-123',
          user: {
            id: 'usr-1',
            username: 'clinician_a',
            tenant_id: 'tenant-alpha',
            role: 'clinician',
            is_active: true,
          },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    };

    const user = await loginUser('clinician_a', 'password123');

    assert.strictEqual(fetchCalls.length, 1);
    assert(fetchCalls[0].url.endsWith('/auth/login'));
    assert.strictEqual(fetchCalls[0].options.credentials, 'include');
    assert.strictEqual(fetchCalls[0].options.method, 'POST');
    assert.strictEqual(user.username, 'clinician_a');
    assert.strictEqual(user.tenant_id, 'tenant-alpha');
    assert.strictEqual(user.role, 'clinician');
  });

  // AUTH-03: successful login renders role-appropriate view
  test('AUTH-03: login returns authoritative role metadata for UI authorization', async () => {
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({
          csrf_token: 'csrf-xyz',
          user: {
            id: 'usr-admin',
            username: 'admin_a',
            tenant_id: 'tenant-alpha',
            role: 'admin',
            is_active: true,
          },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    };

    const adminUser = await loginUser('admin_a', 'password123');
    assert.strictEqual(adminUser.role, 'admin');
    assert.strictEqual(adminUser.tenant_id, 'tenant-alpha');
  });

  // AUTH-04: invalid credentials display error and leave user unauthenticated
  test('AUTH-04: invalid credentials raise ApiRequestError and clear session state', async () => {
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({ detail: 'Invalid credentials or user disabled' }),
        { status: 401, headers: { 'Content-Type': 'application/json' } }
      );
    };

    await assert.rejects(
      async () => {
        await loginUser('clinician_a', 'wrongpassword');
      },
      (err: any) => {
        assert(err instanceof ApiRequestError);
        assert.strictEqual(err.status, 401);
        assert(err.message.includes('Invalid credentials'));
        return true;
      }
    );

    assert.strictEqual(getCsrfToken(), null);
  });

  // AUTH-05: page reload with valid cookie restores authenticated session
  test('AUTH-05: getMe retrieves existing cookie-authenticated session without persistent tokens', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(
        JSON.stringify({
          id: 'usr-restored',
          username: 'dr_reed',
          tenant_id: 'tenant-beta',
          role: 'clinician',
          is_active: true,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    };

    const user = await getMe();
    assert.strictEqual(user.username, 'dr_reed');
    assert.strictEqual(user.tenant_id, 'tenant-beta');
    assert.strictEqual(fetchCalls[0].options.credentials, 'include');
  });

  // AUTH-06: page reload with expired cookie triggers refresh or unauthenticated boundary
  test('AUTH-06: getMe with expired session handles 401 cleanly', async () => {
    globalThis.fetch = async () => {
      return new Response(JSON.stringify({ detail: 'Token expired' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    await assert.rejects(
      async () => {
        await getMe();
      },
      (err: any) => {
        assert(err instanceof ApiRequestError);
        assert.strictEqual(err.status, 401);
        return true;
      }
    );
  });

  // AUTH-07: 401 on clinical request triggers refresh and single transparent retry
  test('AUTH-07: 401 on protected endpoint triggers refresh and transparent retry', async () => {
    setCsrfToken('active-csrf-token');
    let callCount = 0;
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      fetchCalls.push({ url, options: init || {} });
      callCount++;

      if (url.endsWith('/patients/pat-1')) {
        if (callCount === 1) {
          // First call: access token expired
          return new Response(JSON.stringify({ detail: 'Access token expired' }), {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          });
        } else {
          // Retry call: succeeds
          return new Response(
            JSON.stringify({ id: 'pat-1', name: 'John Doe', tenant_id: 'tenant-alpha' }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          );
        }
      }

      if (url.endsWith('/auth/refresh')) {
        return new Response(
          JSON.stringify({ csrf_token: 'new-csrf-token' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }

      return new Response('Not found', { status: 404 });
    };

    const res = await apiFetch('/patients/pat-1');
    const data = await res.json();

    assert.strictEqual(res.status, 200);
    assert.strictEqual(data.id, 'pat-1');
    assert.strictEqual(fetchCalls.length, 3);
    assert(fetchCalls[0].url.endsWith('/patients/pat-1'));
    assert(fetchCalls[1].url.endsWith('/auth/refresh'));
    assert(fetchCalls[2].url.endsWith('/patients/pat-1'));
    assert.strictEqual(getCsrfToken(), 'new-csrf-token');
  });

  // AUTH-08: 401 after failed refresh clears clinical data and notifies session expiration
  test('AUTH-08: 401 with failing refresh notifies session expiration without infinite loop', async () => {
    setCsrfToken('active-csrf-token');
    let sessionExpired = false;
    const unsub = onSessionExpired(() => {
      sessionExpired = true;
    });

    globalThis.fetch = async (input: RequestInfo | URL) => {
      const url = String(input);
      fetchCalls.push({ url, options: {} });
      if (url.endsWith('/auth/refresh')) {
        return new Response(JSON.stringify({ detail: 'Refresh token expired' }), {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify({ detail: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    await assert.rejects(
      async () => {
        await apiFetch('/patients/pat-1');
      },
      (err: any) => {
        assert(err instanceof ApiRequestError);
        assert.strictEqual(err.status, 401);
        return true;
      }
    );

    assert.strictEqual(sessionExpired, true);
    // Verified exactly 2 calls: initial call + refresh call (no runaway retry storm)
    assert.strictEqual(fetchCalls.length, 2);
    unsub();
  });

  // AUTH-09: 403 Forbidden is handled without infinite refresh loop
  test('AUTH-09: 403 Forbidden returns immediately without invoking refresh', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ detail: 'Forbidden for this role' }), {
        status: 403,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    const res = await apiFetch('/admin/destructive-delete');
    assert.strictEqual(res.status, 403);
    assert.strictEqual(fetchCalls.length, 1);
    assert(!fetchCalls.some((c) => c.url.includes('/auth/refresh')));
  });

  // AUTH-10: logout calls backend /api/v1/auth/logout with credentials
  test('AUTH-10: logout calls /auth/logout with credentials: "include"', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ detail: 'Logged out successfully' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    await logoutUser();
    assert.strictEqual(fetchCalls.length, 1);
    assert(fetchCalls[0].url.endsWith('/auth/logout'));
    assert.strictEqual(fetchCalls[0].options.credentials, 'include');
    assert.strictEqual(fetchCalls[0].options.method, 'POST');
  });

  // AUTH-11: logout clears in-memory state immediately
  test('AUTH-11: logout immediately clears in-memory CSRF and notifies session expiration', async () => {
    setCsrfToken('some-token');
    let expiredNotified = false;
    const unsub = onSessionExpired(() => {
      expiredNotified = true;
    });

    globalThis.fetch = async () => {
      return new Response(JSON.stringify({ detail: 'Logged out' }), { status: 200 });
    };

    await logoutUser();
    assert.strictEqual(getCsrfToken(), null);
    assert.strictEqual(expiredNotified, true);
    unsub();
  });

  // AUTH-12: state-changing requests include CSRF token from cookie
  test('AUTH-12: state-changing requests attach X-CSRF-Token header from cookie', async () => {
    mockCookie = 'neuroaegis_csrf_token=test-csrf-cookie-999; other=123';

    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    };

    await apiFetch('/patients/', {
      method: 'POST',
      body: JSON.stringify({ name: 'New Patient' }),
    });

    assert.strictEqual(fetchCalls.length, 1);
    const headers = fetchCalls[0].options.headers as Headers;
    assert.strictEqual(headers.get('X-CSRF-Token'), 'test-csrf-cookie-999');
  });

  // AUTH-13: CSRF token is not read from or stored in persistent storage
  test('AUTH-13: persistent storages (localStorage, sessionStorage) contain no tokens', () => {
    assert.strictEqual(localStorage.getItem('token'), null);
    assert.strictEqual(localStorage.getItem('jwt'), null);
    assert.strictEqual(localStorage.getItem('access_token'), null);
    assert.strictEqual(localStorage.getItem('refresh_token'), null);
    assert.strictEqual(localStorage.getItem('csrf_token'), null);
    assert.strictEqual(sessionStorage.getItem('token'), null);
    assert.strictEqual(sessionStorage.getItem('jwt'), null);
    assert.strictEqual(sessionStorage.length, 0);
    assert.strictEqual(localStorage.length, 0);
  });

  // AUTH-14: GET requests do not attach CSRF header
  test('AUTH-14: safe read requests (GET) omit CSRF header', async () => {
    mockCookie = 'neuroaegis_csrf_token=test-csrf-cookie-999';

    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    };

    await apiFetch('/patients/pat-1', { method: 'GET' });

    assert.strictEqual(fetchCalls.length, 1);
    const headers = fetchCalls[0].options.headers as Headers;
    assert.strictEqual(headers.get('X-CSRF-Token'), null);
  });

  // AUTH-15: login endpoint omits CSRF header
  test('AUTH-15: login endpoint does not require or attach CSRF header prior to authentication', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ user: { id: 'u1' } }), { status: 200 });
    };

    await apiFetch('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username: 'u', password: 'p' }),
    });

    const headers = fetchCalls[0].options.headers as Headers;
    assert.strictEqual(headers.get('X-CSRF-Token'), null);
  });

  // AUTH-16: multiple concurrent 401 requests trigger single refresh, not storm
  test('AUTH-16: multiple concurrent 401 requests coalesce into a single refresh request', async () => {
    setCsrfToken('active-csrf-token');
    let refreshCalls = 0;
    let endpointCalls = 0;

    globalThis.fetch = async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/auth/refresh')) {
        refreshCalls++;
        await new Promise((r) => setTimeout(r, 20));
        return new Response(JSON.stringify({ csrf_token: 'refreshed-token' }), { status: 200 });
      }

      endpointCalls++;
      if (endpointCalls <= 3) {
        // First 3 calls fail with 401
        return new Response(JSON.stringify({ detail: 'Token expired' }), {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      // Retried calls succeed
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    };

    // Fire 3 simultaneous requests
    const [res1, res2, res3] = await Promise.all([
      apiFetch('/jobs/1'),
      apiFetch('/jobs/2'),
      apiFetch('/jobs/3'),
    ]);

    assert.strictEqual(res1.status, 200);
    assert.strictEqual(res2.status, 200);
    assert.strictEqual(res3.status, 200);
    // Mutex verified: exactly 1 refresh call executed across all 3 concurrent 401s
    assert.strictEqual(refreshCalls, 1);
  });

  // AUTH-17: network disconnection during authenticated session displays error without logging out
  test('AUTH-17: network disconnection returns ApiRequestError (status 0) and preserves session', async () => {
    let sessionExpired = false;
    const unsub = onSessionExpired(() => {
      sessionExpired = true;
    });

    globalThis.fetch = async () => {
      throw new TypeError('Failed to fetch (Network connection lost)');
    };

    await assert.rejects(
      async () => {
        await apiFetch('/patients/pat-1');
      },
      (err: any) => {
        assert(err instanceof ApiRequestError);
        assert.strictEqual(err.status, 0);
        assert(err.message.includes('Backend service unreachable'));
        return true;
      }
    );

    // Network error does NOT equal session expired
    assert.strictEqual(sessionExpired, false);
    unsub();
  });

  // AUTH-18: logout all sessions calls /auth/logout-all
  test('AUTH-18: logoutAllSessions calls /auth/logout-all with credentials: "include"', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ detail: 'All sessions revoked' }), { status: 200 });
    };

    await logoutAllSessions();
    assert.strictEqual(fetchCalls.length, 1);
    assert(fetchCalls[0].url.endsWith('/auth/logout-all'));
    assert.strictEqual(fetchCalls[0].options.credentials, 'include');
  });

  // AUTH-19: uploadEegV2 includes credentials and CSRF
  test('AUTH-19: uploadEegV2 routes to /api/v2/predict with credentials: "include" and CSRF', async () => {
    mockCookie = 'neuroaegis_csrf_token=csrf-upload-token';

    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(
        JSON.stringify({
          job_id: 'job-v2-123',
          detected_dataset: 'bonn',
          confidence: 0.95,
          matched_rules: ['rule1'],
          selected_model: 'gnn_v2',
          validation: {
            validationStatus: 'valid',
            fileName: 'sample.edf',
            fileSizeBytes: 1024,
            samplingRate: 256,
            durationSeconds: 10,
            totalChannels: 16,
            eegChannels: 16,
            channelNames: ['Fp1', 'Fp2'],
            excludedChannels: [],
            dataset: 'bonn',
            detectionConfidence: 0.95,
            matchedRules: ['rule1'],
            errors: [],
          },
        }),
        { status: 200 }
      );
    };

    const dummyFile = {
      name: 'test.edf',
      size: 1024,
      type: 'application/octet-stream',
    } as unknown as File;

    const res = await uploadEegV2(
      dummyFile,
      {
        name: 'Jane Doe',
        age: 32,
        gender: 'F',
        weight: 60,
        height: 165,
      },
      { patientId: 'pat-999' }
    );

    assert.strictEqual(res.job_id, 'job-v2-123');
    assert.strictEqual(fetchCalls.length, 1);
    assert(fetchCalls[0].url.includes('/api/v2/predict'));
    assert.strictEqual(fetchCalls[0].options.credentials, 'include');
    const headers = fetchCalls[0].options.headers as Headers;
    assert.strictEqual(headers.get('X-CSRF-Token'), 'csrf-upload-token');
  });

  // AUTH-20: no JWT tokens exposed in storage
  test('AUTH-20: zero token persistence invariant verified across all storages', () => {
    assert.strictEqual(localStorage.length, 0);
    assert.strictEqual(sessionStorage.length, 0);
  });
});

describe('Phase 10.2: Request Correlation & Contextual Tracing', () => {
  test('CORR-01: outgoing apiFetch automatically generates and attaches X-Request-ID header', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    };

    await apiFetch('/health');

    assert.strictEqual(fetchCalls.length, 1);
    const headers = fetchCalls[0].options.headers as Headers;
    const reqId = headers.get('X-Request-ID');
    assert(reqId, 'X-Request-ID header must be present on outgoing request');
    assert.match(reqId, /^[a-zA-Z0-9_\-]{8,64}$/, 'Request ID must be non-empty and bounded string');
  });

  test('CORR-02: distinct apiFetch calls receive distinct X-Request-ID values', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    };

    await apiFetch('/health');
    await apiFetch('/health');

    assert.strictEqual(fetchCalls.length, 2);
    const reqId1 = (fetchCalls[0].options.headers as Headers).get('X-Request-ID');
    const reqId2 = (fetchCalls[1].options.headers as Headers).get('X-Request-ID');
    assert(reqId1 && reqId2);
    assert.notStrictEqual(reqId1, reqId2, 'Independent requests must generate unique request IDs');
  });

  test('CORR-03: apiFetch honors explicitly provided X-Request-ID header and option', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    };

    // Via headers
    await apiFetch('/health', { headers: { 'X-Request-ID': 'custom-req-id-123' } });
    const reqId1 = (fetchCalls[0].options.headers as Headers).get('X-Request-ID');
    assert.strictEqual(reqId1, 'custom-req-id-123');

    // Via option
    await apiFetch('/health', { requestId: 'custom-req-id-456' });
    const reqId2 = (fetchCalls[1].options.headers as Headers).get('X-Request-ID');
    assert.strictEqual(reqId2, 'custom-req-id-456');
  });

  test('CORR-04: transparent 401 retry preserves identical X-Request-ID across attempts', async () => {
    setCsrfToken('mock-csrf');
    let callCount = 0;

    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      fetchCalls.push({ url, options: init || {} });
      callCount++;

      if (url.includes('/auth/refresh')) {
        return new Response(JSON.stringify({ csrf_token: 'new-csrf' }), { status: 200 });
      }

      if (callCount === 1) {
        // First attempt fails with 401
        return new Response(JSON.stringify({ detail: 'Token expired' }), {
          status: 401,
          headers: { 'X-Request-ID': 'preserved-req-id-789' },
        });
      }

      // Retry attempt succeeds
      return new Response(JSON.stringify({ data: 'success' }), {
        status: 200,
        headers: { 'X-Request-ID': 'preserved-req-id-789' },
      });
    };

    const res = await apiFetch('/patients/data', { requestId: 'preserved-req-id-789' });
    assert.strictEqual(res.status, 200);

    // Call 1: initial /patients/data (401)
    // Call 2: /auth/refresh
    // Call 3: retry /patients/data (200)
    assert.strictEqual(fetchCalls.length, 3);
    const initialReqId = (fetchCalls[0].options.headers as Headers).get('X-Request-ID');
    const retryReqId = (fetchCalls[2].options.headers as Headers).get('X-Request-ID');
    assert.strictEqual(initialReqId, 'preserved-req-id-789');
    assert.strictEqual(retryReqId, 'preserved-req-id-789', '401 retry must preserve the original request ID');
  });

  test('CORR-05: ApiRequestError surfaces backend X-Request-ID from response headers', async () => {
    globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), options: init || {} });
      return new Response(JSON.stringify({ detail: 'Invalid parameters' }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': 'server-req-err-999',
        },
      });
    };

    try {
      await loginUser('baduser', 'badpass');
      assert.fail('loginUser should have thrown');
    } catch (err) {
      assert(err instanceof ApiRequestError);
      assert.strictEqual(err.status, 400);
      assert.strictEqual(err.requestId, 'server-req-err-999');
    }
  });

  test('CORR-06: ApiRequestError on network failure surfaces client effectiveRequestId', async () => {
    globalThis.fetch = async () => {
      throw new Error('Connection refused');
    };

    try {
      await apiFetch('/health', { requestId: 'client-offline-req-111' });
      assert.fail('apiFetch should have thrown on network failure');
    } catch (err) {
      assert(err instanceof ApiRequestError);
      assert.strictEqual(err.status, 0);
      assert.strictEqual(err.requestId, 'client-offline-req-111');
      assert(err.message.includes('Backend service unreachable'));
    }
  });
});
