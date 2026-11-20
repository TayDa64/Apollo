#!/usr/bin/env node
/*
 * Scoped cleanup for the MUSE desktop app/server workflow.
 *
 * Avoids the old `taskkill /IM python.exe` behavior, which could kill unrelated
 * Python work on the user's machine.  We stop the JUCE app by image name and
 * stop only the process currently owning the OSC UDP receive port.
 */

const childProcess = require('child_process');

const recvPort = Number(process.env.MMG_OSC_PORT || 9000);
const appImageName = 'AI Music Generator.exe';

function run(command, args, options = {}) {
  const result = childProcess.spawnSync(command, args, {
    encoding: 'utf8',
    windowsHide: true,
    shell: false,
    ...options,
  });

  return {
    status: result.status ?? 0,
    stdout: result.stdout || '',
    stderr: result.stderr || '',
    error: result.error,
  };
}

function stopApp() {
  const result = run('taskkill', ['/IM', appImageName, '/F']);
  const output = `${result.stdout}${result.stderr}`.trim();

  if (result.status === 0) {
    console.log(`Stopped ${appImageName}.`);
    if (output) console.log(output);
    return;
  }

  if (/not found|no tasks are running/i.test(output)) {
    console.log(`${appImageName} was not running.`);
    return;
  }

  console.warn(`Could not stop ${appImageName}: ${output || result.error?.message || 'unknown error'}`);
}

function pidsOwningUdpPort(port) {
  const result = run('netstat', ['-ano', '-p', 'udp']);
  if (result.status !== 0 && !result.stdout) {
    console.warn(`Could not inspect UDP ports: ${result.stderr || result.error?.message || 'netstat failed'}`);
    return [];
  }

  const pids = new Set();
  const portSuffix = `:${port}`;
  for (const line of result.stdout.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.toUpperCase().startsWith('UDP')) continue;

    const parts = trimmed.split(/\s+/);
    if (parts.length < 3) continue;

    const localAddress = parts[1] || '';
    const pid = parts[parts.length - 1];
    if (localAddress.endsWith(portSuffix) && /^\d+$/.test(pid)) {
      pids.add(pid);
    }
  }

  return [...pids];
}

function stopUdpPortOwner(port) {
  const pids = pidsOwningUdpPort(port);
  if (pids.length === 0) {
    console.log(`No process is listening on UDP port ${port}.`);
    return;
  }

  for (const pid of pids) {
    const result = run('taskkill', ['/PID', pid, '/F']);
    const output = `${result.stdout}${result.stderr}`.trim();
    if (result.status === 0) {
      console.log(`Stopped UDP ${port} owner PID ${pid}.`);
      if (output) console.log(output);
    } else if (/not found|no running instance/i.test(output)) {
      console.log(`UDP ${port} owner PID ${pid} was already stopped.`);
    } else {
      console.warn(`Could not stop UDP ${port} owner PID ${pid}: ${output || result.error?.message || 'unknown error'}`);
    }
  }
}

stopApp();
stopUdpPortOwner(recvPort);
