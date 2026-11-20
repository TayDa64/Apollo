#!/usr/bin/env node
/*
 * Start the Python OSC server and JUCE app without creating duplicate server
 * instances.  Duplicate servers crash with WinError 10048 because UDP port 9000
 * is already bound; in that case we reuse the existing server and launch only
 * the app.
 */

const fs = require('fs');
const path = require('path');
const childProcess = require('child_process');

const repoRoot = path.resolve(__dirname, '..');
const recvHost = process.env.MMG_OSC_HOST || '127.0.0.1';
const recvPort = Number(process.env.MMG_OSC_PORT || 9000);
const serverReadyTimeoutMs = Number(process.env.MMG_SERVER_READY_TIMEOUT_MS || 45000);
const pythonExe = process.env.PYTHON || 'python';
const appExe = path.join(repoRoot, 'juce', 'build', 'MultimodalMusicGen_artefacts', 'Release', 'AI Music Generator.exe');

function runtimeLogPath() {
  const runtimeDir = path.join(repoRoot, 'output', '_runtime');
  fs.mkdirSync(runtimeDir, { recursive: true });
  return path.join(runtimeDir, 'mmg-server.log');
}

function startDetachedProcess(command, args, options = {}) {
  const child = childProcess.spawn(command, args, {
    cwd: repoRoot,
    detached: true,
    stdio: options.stdio || 'ignore',
    windowsHide: options.windowsHide ?? false,
    shell: false,
  });
  child.unref();
  return child;
}

function isAppAlreadyRunning() {
  const result = childProcess.spawnSync('tasklist', ['/FI', 'IMAGENAME eq AI Music Generator.exe'], {
    encoding: 'utf8',
    windowsHide: true,
    shell: false,
  });

  if (result.error || result.status !== 0)
    return false;

  return /AI Music Generator\.exe/i.test(result.stdout || '');
}

function startServerWindow() {
  const logPath = runtimeLogPath();
  const logFd = fs.openSync(logPath, 'w');

  console.log(`Starting MMG OSC server on ${recvHost}:${recvPort}...`);
  console.log(`Server log: ${logPath}`);

  try {
    startDetachedProcess(
      pythonExe,
      ['-m', 'multimodal_gen.server', '--host', recvHost, '--port', String(recvPort), '--verbose'],
      { stdio: ['ignore', logFd, logFd], windowsHide: true },
    );
  } catch (err) {
    console.error(`Failed to launch Python server: ${err ? err.message : 'unknown error'}`);
    fs.closeSync(logFd);
    process.exit(1);
  }

  fs.closeSync(logFd);
}

function startApp() {
  if (isAppAlreadyRunning()) {
    console.log('AI Music Generator app already appears to be running; not launching a duplicate window.');
    console.log('If the existing app is frozen, run `npm run stop` first, then retry `npm run start:all`.');
    return;
  }

  if (!fs.existsSync(appExe)) {
    console.error(`App executable not found: ${appExe}`);
    console.error('Run `npm run build` first.');
    process.exit(1);
  }

  console.log('Starting AI Music Generator app...');
  try {
    startDetachedProcess(appExe, [], { stdio: 'ignore', windowsHide: false });
  } catch (err) {
    console.error(`Failed to launch app: ${err ? err.message : 'unknown error'}`);
    process.exit(1);
  }
}

function udpPortOwners(port) {
  const result = childProcess.spawnSync('netstat', ['-ano', '-p', 'udp'], {
    encoding: 'utf8',
    windowsHide: true,
    shell: false,
  });

  if (result.error || (result.status !== 0 && !result.stdout)) {
    console.warn(`Could not inspect UDP ports: ${result.stderr || result.error?.message || 'netstat failed'}`);
    return [];
  }

  const pids = new Set();
  const portSuffix = `:${port}`;
  for (const line of (result.stdout || '').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.toUpperCase().startsWith('UDP')) continue;

    const parts = trimmed.split(/\s+/);
    if (parts.length < 3) continue;

    const localAddress = parts[1] || '';
    const pid = parts[parts.length - 1];
    if (localAddress.endsWith(portSuffix) && /^\d+$/.test(pid))
      pids.add(pid);
  }

  return [...pids];
}

function isUdpPortFree(_host, port) {
  return Promise.resolve(udpPortOwners(port).length === 0);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForUdpPortInUse(host, port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (!(await isUdpPortFree(host, port)))
      return true;

    await delay(250);
  }

  return false;
}

(async () => {
  if (process.platform !== 'win32') {
    console.error('scripts/start-all.js currently targets Windows start/cmd semantics.');
    process.exit(1);
  }

  const free = await isUdpPortFree(recvHost, recvPort);
  if (free) {
    startServerWindow();
    const ready = await waitForUdpPortInUse(recvHost, recvPort, serverReadyTimeoutMs);
    if (ready) {
      console.log(`MMG OSC server is listening on ${recvHost}:${recvPort}.`);
    } else {
      console.error(`MMG OSC server did not bind ${recvHost}:${recvPort} within ${serverReadyTimeoutMs}ms.`);
      console.error('Not launching the app because it would start disconnected/unusable. Check the MMG Server window or run `npm run stop` and retry.');
      process.exit(2);
    }
  } else {
    console.log(`MMG OSC server already appears to be running on ${recvHost}:${recvPort}; reusing it.`);
  }

  startApp();
})();
