# Product Requirements Document
## Quip Network Testnet Light Client

**Version:** 1.0
**Date:** February 16, 2026
**Status:** Draft

---

## Executive Summary

A minimal cross-platform desktop application that enables non-technical users to run Quip Network testnet nodes. The app packages the node software into a single installable binary with a simple UI showing connection status and terminal output.

**Goal:** Maximize testnet participation for network stress testing ahead of TGE (late April/early May 2026).

---

## Problem Statement

Current testnet setup requires Docker, CLI commands, manual port forwarding configuration, and IP detection. This limits participation to technical users. We need broader participation for comprehensive stress testing.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Installation completion rate | >80% |
| Time to connected node | <10 minutes (excluding port forwarding) |
| Concurrent testnet nodes | 500+ during stress test |

---

## Core Requirements

### What This App Does

1. **Runs a testnet node** with one click
2. **Shows connection status** (Connected / Syncing / Disconnected / Stopped)
3. **Displays terminal output** streaming from the node process
4. **Generates and displays node secret** for future participation claims
5. **Guides port forwarding** with test button

### What This App Does NOT Do (MVP)

- Track uptime or participation metrics
- Offer GPU acceleration toggle
- Provide mining rewards (testnet has none)
- Store data server-side

---

## User Flow

### First Launch

1. User downloads installer from website
2. Runs installer, app launches
3. **Setup Wizard:**
   - Welcome screen explaining testnet participation
   - Port forwarding guidance with [Test Connection] button
   - Node secret generated and displayed with **strong prompt to back it up**
4. User clicks [Start Node]
5. Terminal output streams, status shows "Syncing" then "Connected"

### Daily Use

1. Open app (or auto-start on boot)
2. Node runs, user sees status + terminal output
3. Minimize to system tray
4. Node runs in background

### Updates

1. App checks for updates on launch
2. If update available, notification appears
3. User clicks [Update Now]
4. App downloads, stops node, replaces binary, restarts

---

## Functional Requirements

### FR-1: Installation

- **Single-file installers:**
  - Windows: `.exe`
  - macOS: `.dmg`
  - Linux: `.AppImage`
- No prerequisites (no Docker, no runtime dependencies)
- Target size: ~150-200MB (includes bundled Python runtime)

### FR-2: Main Window

```
┌─────────────────────────────────────────────────────┐
│  [Quip Logo]  Quip Network Testnet Node      [—][×] │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Status: ● Connected                                │
│  Public IP: 203.0.113.45:20049                      │
│                                                     │
│  Node Secret: ●●●●●●●●●●●●  [Show] [Copy]          │
│  ⚠️ Save this! Required for future participation    │
│     claims. No way to recover if lost.              │
│                                                     │
│  [ Start Node ]  [ Stop Node ]  [ Restart ]         │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Terminal Output                          [Clear]   │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [2026-02-16 10:23:45] Connecting to network...  │ │
│ │ [2026-02-16 10:23:46] Peer discovered: 4 nodes  │ │
│ │ [2026-02-16 10:23:47] Syncing block 12847...    │ │
│ │ [2026-02-16 10:23:48] Block validated           │ │
│ │ [2026-02-16 10:23:49] Connected to network      │ │
│ │ █                                               │ │
│ └─────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│  v0.1.0                    [Check for Updates]      │
└─────────────────────────────────────────────────────┘
```

### FR-3: Connection Status

| State | Indicator | Description |
|-------|-----------|-------------|
| Connected | 🟢 Green | Node synced and communicating |
| Syncing | 🟡 Yellow | Node catching up |
| Disconnected | 🔴 Red | Cannot connect (show reason) |
| Stopped | ⚫ Gray | Node intentionally stopped |

### FR-4: Node Secret

- Generated locally on first launch
- Displayed masked by default with [Show] and [Copy] buttons
- **Critical UX:** Prominent warning that this must be backed up
- Stored locally in app data directory
- Required for future participation claim (1-2 months post-testnet)

### FR-5: Port Forwarding

- **Hard requirement** - node cannot participate without it
- Auto-detect public IP
- Default port: 20049 (UDP required)
- [Test Connection] button to verify port is reachable
- General guidance + links to common router brand instructions
- If firewall software detected, remind user to allow the port

### FR-6: Terminal Output

- Stream stdout/stderr from node process
- Auto-scroll (with ability to scroll back)
- Basic highlighting (errors red, success green)
- [Copy Logs] button for support
- [Clear] button
- Persist last 10,000 lines to disk

### FR-7: Update System

- Check for updates on launch + every 6 hours
- Display notification when update available
- One-click update: download → stop node → replace binary → restart
- Manual [Check for Updates] button

### FR-8: System Tray

- Minimize to tray (configurable)
- Tray icon shows connection status color
- Right-click menu: Show Window, Start/Stop, Quit

### FR-9: Settings

Minimal settings panel:
- Public IP (auto-detected, editable)
- Port number (default 20049)
- Start on system boot (checkbox)
- Minimize to tray on close (checkbox)

### FR-10: Error Handling

| Error | Message | Action |
|-------|---------|--------|
| Port closed | "Cannot connect. Port 20049 not reachable." | Link to guide, [Test] button |
| No internet | "No internet connection." | [Retry] button |
| Node crash | "Node stopped unexpectedly." | [Restart], [Copy Logs] |

---

## Non-Functional Requirements

### Compatibility

| Platform | Minimum Version | Architecture |
|----------|-----------------|--------------|
| Windows | Windows 10 | x64 |
| macOS | macOS 11 (Big Sur) | x64, ARM64 |
| Linux | Ubuntu 20.04+ | x64 |

### Performance

- App startup: <3 seconds
- UI memory: <200MB
- Install size: ~150-200MB (bundled Python + dependencies)

### Security

- Code signed (Windows + macOS)
- Update downloads verified via checksum
- No telemetry without consent

---

## Technical Architecture

### Stack: Tauri + PyInstaller

The node is a **Python application** (`quip-protocol` package). We extract it from the Docker image and bundle it with PyInstaller into a standalone executable.

**Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Tauri Application                         │
├──────────────────────┬──────────────────────────────────────┤
│   Frontend (Web)     │         Backend (Rust)               │
├──────────────────────┼──────────────────────────────────────┤
│   • Status UI        │  • Node process management           │
│   • Terminal view    │  • Log streaming via stdout/stderr   │
│   • Settings forms   │  • Config file management            │
│   • Node secret      │  • Update checker/downloader         │
│     display          │  • System tray integration           │
│                      │  • Port test (external service)      │
└──────────────────────┴──────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  quip-node          │
                    │  (PyInstaller exe)  │
                    ├─────────────────────┤
                    │  Python 3.13        │
                    │  + quip_cli.py      │
                    │  + shared/          │
                    │  + CPU/             │
                    │  + 50+ dependencies │
                    └─────────────────────┘
```

### Node Binary Creation (No Engineering Team Needed)

We extract the Python code from the existing Docker image and bundle it:

```bash
# 1. Extract from Docker
docker cp quip-cpu:/usr/local/lib/python3.13/site-packages/quip_cli.py .
docker cp quip-cpu:/usr/local/lib/python3.13/site-packages/shared .
docker cp quip-cpu:/usr/local/lib/python3.13/site-packages/CPU .
docker cp quip-cpu:/app/genesis_block_public.json .

# 2. Bundle with PyInstaller (per platform)
pyinstaller --onefile --name quip-node quip_cli.py
```

**Cross-platform builds:**
- macOS: Build on macOS (x64 + ARM64 via universal2)
- Windows: Build on Windows or via Wine/cross-compilation
- Linux: Build on Linux (x64)

### Node Secret Implementation

The node secret is a **64-character hex string** generated on first run:

```python
# Generated via: openssl rand -hex 32
secret = "a1b2c3d4e5f6..."  # 64 hex chars
```

Stored in `~/quip-data/config.toml`:
```toml
[global]
secret = "a1b2c3d4e5f6..."
```

**App must:**
1. Generate secret on first launch (if config.toml doesn't exist)
2. Display masked with [Show] [Copy] buttons
3. Warn user prominently to back it up

### Data Storage

```
~/quip-data/
├── config.toml           # Node config (contains secret!)
├── trust.db              # Certificate trust store
├── app-settings.json     # UI preferences (separate from node)
├── certs/                # TLS certificates
└── logs/
    └── app.log           # Application logs (not node logs)
```

**Note:** Node logs stream to stdout/stderr, captured by Tauri, not written to disk by default.

### Key Dependencies (from Docker image)

Core packages required for the node:
- `aioquic` - QUIC protocol for P2P networking
- `aiohttp` - Async HTTP client
- `blake3` - Fast hashing
- `cryptography` - TLS/crypto operations
- `dwave-ocean-sdk` - Quantum annealing simulation (CPU mode)
- `numpy`, `scipy` - Numerical computing

Full list: 60+ packages, ~462MB uncompressed, ~150-200MB bundled

---

## Dependencies & Status

| Item | Owner | Status |
|------|-------|--------|
| Node code extraction | ✅ Done | Extracted from Docker |
| PyInstaller builds | Build | 🔄 To be set up |
| Code signing certificates | DevOps | ❓ Unknown |
| Update server / CDN | DevOps | ❓ Unknown |
| Port test endpoint | Backend | ❓ Unknown (can use external) |
| Logo / icons | Design | 📁 Ready for upload |
| Discord link | Community | 📝 Pending |

**No blockers remaining.** Node binary can be created from extracted Python code.

---

## MVP Scope

### In Scope

- [x] Single-window app with terminal output
- [x] Start/Stop/Restart node
- [x] Connection status indicator
- [x] Node secret generation + display + backup warning
- [x] Port forwarding guidance + test button
- [x] One-click updates
- [x] System tray
- [x] Windows, macOS, Linux builds

### Out of Scope (Post-MVP)

- Uptime tracking / participation metrics
- GPU acceleration toggle
- Advanced settings
- In-app analytics
- Localization

---

## Timeline

| Week | Milestone |
|------|-----------|
| Week 1 | PyInstaller builds for all platforms, Tauri project setup, basic UI shell |
| Week 2 | Node process spawning, log streaming, status detection, start/stop controls |
| Week 3 | Setup wizard, port guidance, node secret display, settings panel |
| Week 4 | Update system, system tray, error handling, cross-platform testing, release |

**No external blockers.** Can begin immediately with extracted node code.

---

## Support & Communication

- **Primary support channel:** Discord (link TBD)
- **In-app:** "Need help? Join our Discord" link
- **Diagnostic info:** [Copy Diagnostic Info] button for support requests

---

## Open Questions

1. ~~Who owns producing standalone node binaries?~~ **RESOLVED:** Extract from Docker + PyInstaller
2. Do we have code signing certificates for Windows/macOS?
3. Where will update binaries be hosted? (GitHub Releases is simplest)
4. Port testing: Use external service (e.g., canyouseeme.org API) or build simple endpoint?
5. Discord invite link for support channel?

---

## Appendix: Port Forwarding Guide

### What is port forwarding?

Your router blocks incoming connections by default. Port forwarding tells your router to allow Quip Network traffic through to your computer.

### Steps

1. Open router admin (usually `192.168.0.1` or `192.168.1.1`)
2. Find "Port Forwarding" settings
3. Add rule:
   - Port: **20049**
   - Protocol: **UDP** (required)
   - Internal IP: Your computer's local IP
4. Save and test

### Using a firewall?

Make sure your firewall software (Windows Defender, etc.) allows incoming connections on port 20049.

### Router guides

- [Netgear](https://www.netgear.com/support/)
- [TP-Link](https://www.tp-link.com/support/)
- [ASUS](https://www.asus.com/support/)
- [Linksys](https://www.linksys.com/support/)

---

*Document ready for review*
