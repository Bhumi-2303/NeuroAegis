import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import assert from 'node:assert';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distDir = path.resolve(__dirname, '../dist');
const testEdfPath = path.resolve(__dirname, '../../../scratch/test_chbmit.edf');

// Simple static file server for dist/
function startServer(port = 4173) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.woff2': 'font/woff2',
  };

  const server = http.createServer((req, res) => {
    let filePath = path.join(distDir, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      filePath = path.join(distDir, 'index.html');
    }

    const ext = path.extname(filePath);
    const contentType = mimeTypes[ext] || 'application/octet-stream';

    try {
      const content = fs.readFileSync(filePath);
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  });

  return new Promise((resolve) => {
    server.listen(port, () => resolve(server));
  });
}

(async () => {
  console.log('================================================================');
  console.log('  PROMPT 9.4.1 BROWSER E2E TEST SUITE (AUTH-E2E-01 to 08)');
  console.log('================================================================');

  const port = 4173;
  const server = await startServer(port);
  console.log(`Static web server running on http://127.0.0.1:${port}`);

  const executablePath = '/Users/tirthkosambia/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';
  let browser;

  try {
    browser = await chromium.launch({
      headless: true,
      executablePath: fs.existsSync(executablePath) ? executablePath : undefined,
    });

    const recordedNetworkRequests = [];
    const recordedConsoleErrors = [];

    // Helper to configure page routes and listeners
    const setupPageContext = async (context) => {
      let currentSessionUser = null; // null | { id, username, role, tenant_id }

      const page = await context.newPage();

      page.on('console', (msg) => {
        if (msg.type() === 'error') {
          recordedConsoleErrors.push(msg.text());
        }
      });

      page.on('pageerror', (err) => {
        recordedConsoleErrors.push(err.message);
      });

      await page.route('**/api/v1/**', async (route) => {
        const req = route.request();
        const url = req.url();
        const method = req.method();
        const headers = req.headers();

        recordedNetworkRequests.push({
          url,
          method,
          headers,
          hasCsrfHeader: Boolean(headers['x-csrf-token'] || headers['X-CSRF-Token']),
          hasAuthBearer: Boolean(headers['authorization']?.toLowerCase().includes('bearer')),
        });

        // /auth/me
        if (url.endsWith('/auth/me')) {
          if (!currentSessionUser) {
            await route.fulfill({
              status: 401,
              contentType: 'application/json',
              body: JSON.stringify({ detail: 'Not authenticated' }),
            });
          } else {
            await route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify(currentSessionUser),
            });
          }
          return;
        }

        // /auth/login
        if (url.endsWith('/auth/login') && method === 'POST') {
          const body = JSON.parse(req.postData() || '{}');
          if (body.password === 'wrongpassword') {
            await route.fulfill({
              status: 401,
              contentType: 'application/json',
              body: JSON.stringify({ detail: 'Invalid credentials or user disabled' }),
            });
            return;
          }

          let role = 'clinician';
          if (body.username.includes('admin')) role = 'admin';
          if (body.username.includes('researcher')) role = 'researcher';

          currentSessionUser = {
            id: `usr-${body.username}`,
            username: body.username,
            tenant_id: 'tenant-omega',
            role,
            is_active: true,
          };

          // Set CSRF cookie (simulating backend set_auth_cookies)
          await context.addCookies([
            {
              name: 'neuroaegis_csrf_token',
              value: 'csrf-session-token-omega',
              url: `http://127.0.0.1:${port}`,
            },
          ]);

          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            headers: {
              'Set-Cookie': 'neuroaegis_csrf_token=csrf-session-token-omega; Path=/; SameSite=Strict',
            },
            body: JSON.stringify({
              csrf_token: 'csrf-session-token-omega',
              user: currentSessionUser,
            }),
          });
          return;
        }

        // /auth/logout
        if (url.endsWith('/auth/logout') && method === 'POST') {
          currentSessionUser = null;
          await context.clearCookies();
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'Logged out successfully' }),
          });
          return;
        }

        // /predict/ (Clinical EEG Upload)
        if (url.includes('/predict') && method === 'POST') {
          if (!currentSessionUser || currentSessionUser.role === 'researcher') {
            await route.fulfill({
              status: 403,
              contentType: 'application/json',
              body: JSON.stringify({ detail: 'Forbidden: Insufficient privileges for clinical prediction' }),
            });
            return;
          }

          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              job_id: 'job-real-001',
              detected_dataset: 'chbmit',
              confidence: 0.98,
              matched_rules: ['chbmit_standard_montage'],
              selected_model: 'lightgbm',
              validation: {
                validationStatus: 'valid',
                fileName: 'test_chbmit.edf',
                fileSizeBytes: 65024,
                samplingRate: 256,
                durationSeconds: 5,
                totalChannels: 21,
                eegChannels: 21,
                channelNames: ['FP1-F7', 'F7-T7'],
                excludedChannels: [],
                dataset: 'chbmit',
                detectionConfidence: 0.98,
                matchedRules: ['chbmit_standard_montage'],
                errors: [],
              },
            }),
          });
          return;
        }

        // /jobs/job-real-001
        if (url.includes('/jobs/job-real-001')) {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              job_id: 'job-real-001',
              status: 'Completed',
              progress: 100,
              datasetName: 'chbmit',
              detectionConfidence: 0.98,
              modelName: 'lightgbm',
              result: {
                prediction_label: 'non_seizure',
                probability_seizure: 0.08,
                confidence_band: 'high',
                shap_explanation: {
                  baseValue: 0.05,
                  features: [
                    { featureName: 'delta_power (FP1-F7)', value: 0.02, rawValue: 1.15, referenceRange: [0.5, 2.0] },
                    { featureName: 'theta_power (F7-T7)', value: 0.01, rawValue: 0.82, referenceRange: [0.3, 1.5] },
                  ],
                },
                eeg_visualization: {
                  dataset: 'chbmit',
                  fileName: 'test_chbmit.edf',
                  fileSizeBytes: 65024,
                  patientIdentifier: 'PAT-CHB01',
                  samplingRate: 256,
                  durationSeconds: 5,
                  totalChannels: 21,
                  eegChannelCount: 21,
                  channels: [
                    { id: 'FP1-F7', name: 'FP1-F7', samples: [0.1, -0.2, 0.4, 0.2, -0.1], samplingRate: 256 },
                    { id: 'F7-T7', name: 'F7-T7', samples: [0.0, 0.1, -0.1, 0.3, 0.0], samplingRate: 256 },
                  ],
                  visualizationSampleCount: 5,
                  originalSampleCount: 1280,
                  timeStartSeconds: 0,
                  timeEndSeconds: 5,
                  seizures: [],
                  hasSeizureAnnotations: false,
                  annotationStatus: 'available',
                  annotationSource: 'automated_detection',
                  referenceAvailable: false,
                  reference: null,
                  channelActivity: [
                    { channelName: 'FP1-F7', activityScore: 1.15, baselineScore: 1.0, relativeChange: 0.15, metrics: {} },
                  ],
                  channelActivityAvailable: true,
                  channelActivityNote: 'Baseline normalized activity',
                  excludedChannels: [],
                  excludedChannelDetails: [],
                },
              },
            }),
          });
          return;
        }

        await route.continue();
      });

      return { page, setSessionUser: (user) => { currentSessionUser = user; } };
    };

    // --------------------------------------------------------------------------
    // AUTH-E2E-01: Session restoration after reload
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-01] Testing session restoration after reload...');
    const ctx1 = await browser.newContext();
    const { page: page1 } = await setupPageContext(ctx1);

    await page1.goto(`http://127.0.0.1:${port}/`);
    await page1.waitForSelector('text=Institutional Session Sign-In');

    // Login as clinician
    await page1.fill('input#username', 'clinician_a');
    await page1.fill('input#password', 'password123');
    await page1.click('button[type="submit"]');

    // Verify clinical dashboard
    await page1.waitForSelector('text=NEUROAEGIS');
    await page1.waitForSelector('text=clinician');
    await page1.waitForSelector('text=[tenant-omega]');
    console.log('  ✓ Initial login successful, clinical dashboard rendered.');

    // Reload browser page
    console.log('  Reloading browser page to test session restoration...');
    await page1.reload();

    // Confirm session restored via /auth/me without asking to log in again
    await page1.waitForSelector('text=NEUROAEGIS');
    await page1.waitForSelector('text=clinician');
    await page1.waitForSelector('text=[tenant-omega]');
    const loginHeaderAfterReload = await page1.$('text=Institutional Session Sign-In');
    assert.strictEqual(loginHeaderAfterReload, null, 'Login form must not be shown on restored session');
    console.log('  ✓ AUTH-E2E-01 PASSED: Session cleanly restored after browser reload.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-02: Direct protected-route navigation while unauthenticated
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-02] Testing direct protected-route navigation while unauthenticated...');
    const ctx2 = await browser.newContext();
    const { page: page2 } = await setupPageContext(ctx2);

    await page2.goto(`http://127.0.0.1:${port}/`);
    await page2.waitForSelector('text=Institutional Session Sign-In');

    const clinicalHeader = await page2.$('text=Real-Time EEG & Seizure Telemetry');
    assert.strictEqual(clinicalHeader, null, 'Clinical header must not be rendered when unauthenticated');
    console.log('  ✓ Unauthenticated user is blocked by authentication boundary.');

    // Log in
    await page2.fill('input#username', 'clinician_a');
    await page2.fill('input#password', 'password123');
    await page2.click('button[type="submit"]');
    await page2.waitForSelector('text=NEUROAEGIS');
    console.log('  ✓ AUTH-E2E-02 PASSED: Protected application accessible after valid login.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-03: Protected route remains inaccessible after logout
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-03] Testing protected route remains inaccessible after logout...');
    // Click Sign Out
    await page2.click('button[aria-label="Sign out of clinical session"]');
    await page2.waitForSelector('text=Institutional Session Sign-In');
    console.log('  ✓ Successfully logged out.');

    // Directly navigate again
    await page2.goto(`http://127.0.0.1:${port}/`);
    await page2.waitForSelector('text=Institutional Session Sign-In');
    const clinicalHeaderPostLogout = await page2.$('text=Real-Time EEG & Seizure Telemetry');
    assert.strictEqual(clinicalHeaderPostLogout, null, 'Clinical header must remain inaccessible post-logout');

    // Storage check
    const storagePostLogout = await page2.evaluate(() => ({
      local: Object.keys(localStorage),
      session: Object.keys(sessionStorage),
    }));
    assert.strictEqual(storagePostLogout.local.length, 0, 'localStorage must be empty');
    assert.strictEqual(storagePostLogout.session.length, 0, 'sessionStorage must be empty');
    console.log('  ✓ AUTH-E2E-03 PASSED: Protected route is inaccessible after logout; storage clean.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-04: Admin role UI boundary
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-04] Testing Admin role UI boundary...');
    const ctx3 = await browser.newContext();
    const { page: page3 } = await setupPageContext(ctx3);

    await page3.goto(`http://127.0.0.1:${port}/`);
    await page3.fill('input#username', 'admin_a');
    await page3.fill('input#password', 'password123');
    await page3.click('button[type="submit"]');

    await page3.waitForSelector('text=NEUROAEGIS');
    await page3.waitForSelector('text=admin');
    await page3.waitForSelector('text=[tenant-omega]');
    // Admin Controls badge should be present for Admin role
    await page3.waitForSelector('text=Admin Controls');
    console.log('  ✓ Admin role verified: Admin Controls indicator and role badge displayed.');
    console.log('  ✓ AUTH-E2E-04 PASSED: Admin role UI boundary verified.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-05: Researcher 403 does not cause logout
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-05] Testing Researcher 403 behavior and PHI boundary...');
    const ctx4 = await browser.newContext();
    const { page: page4 } = await setupPageContext(ctx4);

    await page4.goto(`http://127.0.0.1:${port}/`);
    await page4.fill('input#username', 'researcher_a');
    await page4.fill('input#password', 'password123');
    await page4.click('button[type="submit"]');

    // Confirm Researcher Portal rendered
    await page4.waitForSelector('text=RESEARCH PORTAL');
    await page4.waitForSelector('text=Clinical PHI & Live Telemetry Access Boundary');
    await page4.waitForSelector('text=Under institutional privacy boundaries and multi-tenant clinical data isolation policy');

    // Simulate an unauthorized clinical request from inside the browser context
    console.log('  Triggering clinical operation from researcher session to observe 403...');
    const responseStatus = await page4.evaluate(async () => {
      try {
        const match = document.cookie.match(/(?:^|;\s*)neuroaegis_csrf_token=([^;]+)/);
        const csrf = match ? decodeURIComponent(match[1]) : '';
        const headers = {};
        if (csrf) headers['X-CSRF-Token'] = csrf;
        const res = await fetch('/api/v1/predict/', {
          method: 'POST',
          headers,
          credentials: 'include',
        });
        return res.status;
      } catch (e) {
        return e?.status || -1;
      }
    });
    assert.strictEqual(responseStatus, 403, 'Endpoint must return 403 Forbidden for researcher');

    // Confirm Researcher remains authenticated and NOT logged out
    await page4.waitForSelector('text=RESEARCH PORTAL');
    const loginFormAfter403 = await page4.$('text=Institutional Session Sign-In');
    assert.strictEqual(loginFormAfter403, null, '403 must NOT redirect or log out the user');
    console.log('  ✓ 403 Forbidden received without triggering logout or redirect.');
    console.log('  ✓ AUTH-E2E-05 PASSED: Researcher 403 does not cause logout.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-06: Authenticated clinical workflow (Upload EDF, Poll, Visualize)
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-06] Testing authenticated clinical workflow (EDF Upload -> Inference -> Visualization)...');
    assert(fs.existsSync(testEdfPath), `Test EDF fixture must exist at ${testEdfPath}`);

    const ctx5 = await browser.newContext();
    const { page: page5 } = await setupPageContext(ctx5);

    await page5.goto(`http://127.0.0.1:${port}/`);
    await page5.fill('input#username', 'clinician_a');
    await page5.fill('input#password', 'password123');
    await page5.click('button[type="submit"]');
    await page5.waitForSelector('text=NEUROAEGIS');

    // Select EDF file
    console.log(`  Uploading EDF fixture (${testEdfPath})...`);
    const fileInput = await page5.waitForSelector('input[type="file"]', { state: 'attached', timeout: 5000 });
    assert(fileInput, 'EDF file input element must be present in Dashboard');
    await fileInput.setInputFiles(testEdfPath);

    // Wait for analysis to complete and results to render
    console.log('  Waiting for prediction job and visualization to render...');
    await page5.waitForSelector('text=REAL EEG ANALYSIS', { timeout: 10000 });
    await page5.waitForSelector('text=PAT-CHB01', { timeout: 10000 });
    await page5.waitForSelector('text=PATIENT EEG - SEIZURE ANALYSIS', { timeout: 10000 });
    await page5.waitForSelector('text=delta_power (FP1-F7)', { timeout: 10000 });

    console.log('  ✓ Real EEG metadata rendered: PAT-CHB01.');
    console.log('  ✓ Waveform canvas updated to PATIENT EEG - SEIZURE ANALYSIS.');
    console.log('  ✓ SHAP feature attribution rendered successfully.');
    console.log('  ✓ AUTH-E2E-06 PASSED: Full authenticated clinical workflow executed end-to-end.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-07: Browser storage audit
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-07] Running browser storage runtime audit across all storages...');
    const storageAudit = await page5.evaluate(() => {
      return {
        localKeys: Object.keys(localStorage),
        sessionKeys: Object.keys(sessionStorage),
        cookieStr: document.cookie,
      };
    });

    assert.strictEqual(storageAudit.localKeys.length, 0, 'localStorage must be empty');
    assert.strictEqual(storageAudit.sessionKeys.length, 0, 'sessionStorage must be empty');
    // Ensure document.cookie does not contain access or refresh tokens
    assert(!storageAudit.cookieStr.includes('neuroaegis_access_token'), 'neuroaegis_access_token must not be in document.cookie');
    assert(!storageAudit.cookieStr.includes('neuroaegis_refresh_token'), 'neuroaegis_refresh_token must not be in document.cookie');
    console.log('  ✓ Storage audit confirmed: 0 items in localStorage/sessionStorage.');
    console.log('  ✓ Cookie audit confirmed: HttpOnly auth tokens are inaccessible to JavaScript.');
    console.log('  ✓ AUTH-E2E-07 PASSED: Browser storage remains completely free of tokens and PHI.');

    // --------------------------------------------------------------------------
    // AUTH-E2E-08: Network request authentication & CSRF contract audit
    // --------------------------------------------------------------------------
    console.log('\n[AUTH-E2E-08] Running network traffic contract audit...');
    assert(recordedNetworkRequests.length > 0, 'Recorded network requests must not be empty');

    let stateChangingCount = 0;
    let safeReadCount = 0;

    for (const req of recordedNetworkRequests) {
      assert(!req.hasAuthBearer, `Request to ${req.url} must NOT use Authorization: Bearer`);
      if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method)) {
        if (!req.url.endsWith('/auth/login')) {
          assert(req.hasCsrfHeader, `State-changing request ${req.method} ${req.url} must include X-CSRF-Token`);
          stateChangingCount++;
        }
      } else if (['GET', 'HEAD'].includes(req.method)) {
        assert(!req.hasCsrfHeader, `Safe read request ${req.method} ${req.url} must NOT send CSRF header`);
        safeReadCount++;
      }
    }

    console.log(`  ✓ Verified ${recordedNetworkRequests.length} total API requests:`);
    console.log(`    - ${stateChangingCount} state-changing requests had X-CSRF-Token header.`);
    console.log(`    - ${safeReadCount} read requests omitted unnecessary CSRF headers.`);
    console.log(`    - 0 requests used Authorization: Bearer JWTs.`);
    console.log('  ✓ AUTH-E2E-08 PASSED: Network request authentication and CSRF contract verified.');

    // Console errors check
    console.log('\nChecking recorded browser console errors...');
    const unexpectedErrors = recordedConsoleErrors.filter(
      (err) => !err.includes('403') && !err.includes('401')
    );
    assert.strictEqual(unexpectedErrors.length, 0, `Unexpected console errors found: ${unexpectedErrors.join(', ')}`);
    console.log('  ✓ Zero unexpected browser console errors.');

    console.log('\n================================================================');
    console.log('  ALL PLAYWRIGHT BROWSER E2E TESTS (AUTH-E2E-01 to 08) PASSED!');
    console.log('================================================================\n');
  } finally {
    if (browser) await browser.close();
    server.close();
  }
})();
