#!/usr/bin/env node
/*
 * Resolve and run CMake without requiring cmake.exe to be on PATH.
 * This keeps `npm run build` reliable on Windows/VS installs and still works
 * when a user installs a local tools/cmake distribution later.
 */

const fs = require('fs');
const path = require('path');
const childProcess = require('child_process');

const repoRoot = path.resolve(__dirname, '..');

function unique(items) {
  return [...new Set(items.filter(Boolean))];
}

function candidatePaths() {
  const envCandidates = [process.env.CMAKE_EXE, process.env.CMAKE_PATH];
  const localCandidates = [
    path.join(repoRoot, 'tools', 'cmake', 'bin', process.platform === 'win32' ? 'cmake.exe' : 'cmake'),
    path.join(repoRoot, '.tools', 'cmake', 'bin', process.platform === 'win32' ? 'cmake.exe' : 'cmake'),
  ];

  const windowsCandidates = process.platform === 'win32' ? [
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe',
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe',
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe',
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe',
    'C:\\Program Files\\CMake\\bin\\cmake.exe',
  ] : [];

  return unique([...envCandidates, ...localCandidates, ...windowsCandidates]);
}

function findOnPath() {
  const command = process.platform === 'win32' ? 'where' : 'which';
  const result = childProcess.spawnSync(command, ['cmake'], { encoding: 'utf8' });
  if (result.status !== 0 || !result.stdout) return null;
  return result.stdout.split(/\r?\n/).map(s => s.trim()).find(Boolean) || null;
}

function resolveCmake() {
  for (const candidate of candidatePaths()) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }

  return findOnPath();
}

function printMissingCmakeHelp() {
  console.error('Unable to locate CMake.');
  console.error('Checked CMAKE_EXE/CMAKE_PATH, local tools/cmake, Visual Studio bundled CMake, and PATH.');
  console.error('Fix options:');
  console.error('  1. Install CMake and ensure cmake is on PATH.');
  console.error('  2. Set CMAKE_EXE to the full cmake executable path.');
  console.error('  3. Place a CMake distribution at MUSE/tools/cmake/bin/cmake.exe.');
}

if (require.main === module) {
  const cmake = resolveCmake();
  if (!cmake) {
    printMissingCmakeHelp();
    process.exit(127);
  }

  if (process.argv.includes('--print')) {
    console.log(cmake);
    process.exit(0);
  }

  const args = process.argv.slice(2);
  const result = childProcess.spawnSync(cmake, args, {
    cwd: repoRoot,
    stdio: 'inherit',
    shell: false,
  });

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }

  process.exit(result.status ?? 0);
}

module.exports = { resolveCmake };
