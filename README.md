# Quip Network Node Runner

Desktop application for running a Quip Network node. Mine blocks, earn testnet participation credit, and help test the network.

<p align="center">
  <img src="assets/screenshot.png" alt="Quip Network Node Runner" width="700">
</p>

> **Alpha Software:** This is early testnet software. Expect bugs.

## Download

| Platform | Download |
|----------|----------|
| macOS (Apple Silicon) | [Download .dmg](../../releases/latest) |

### Installation

1. Download and open the `.dmg` file
2. Drag the app to Applications
3. **Important:** The app is not code-signed. macOS will show "damaged" warning. Run this in Terminal:
   ```bash
   xattr -cr "/Applications/Quip Network Node Runner.app"
   ```
4. Now you can open the app normally

## Requirements

- **Port forwarding:** UDP port `20049` must be forwarded to your machine
- **macOS:** macOS 11 (Big Sur) or later

## Quick Start

1. Download and install the app
2. Launch "Quip Network Node Runner"
3. Save your node secret somewhere safe (shown on first launch)
4. Click "Start" to begin mining

## Port Forwarding

The app needs incoming connections on UDP port 20049. To configure:

1. Log into your router (usually `192.168.1.1`)
2. Find Port Forwarding / NAT / Gaming settings
3. Add a rule: External `20049` → Internal `20049` (UDP)
4. Point it to your computer's local IP

## Building from Source

### Prerequisites

- [Node.js](https://nodejs.org/) 20+
- [Rust](https://rustup.rs/) 1.70+

### Build

```bash
git clone https://github.com/rdyplayerB/quip-network-node-runner.git
cd quip-network-node-runner
npm install
npm run bundle-python
npm run tauri build
```

## Development

```bash
npm run tauri dev
```

## License

AGPL-3.0
