//! Node process management.
//!
//! Handles starting, stopping, and monitoring the quip-node process.
//! Uses embedded Python interpreter to run the node.

use crate::{AppConfig, NodeStatus};
use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;
use tauri::{Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

/// Manages the quip-node child process
pub struct NodeManager {
    process: Option<Child>,
    status: NodeStatus,
    stop_signal: Option<mpsc::Sender<()>>,
}

impl NodeManager {
    pub fn new() -> Self {
        Self {
            process: None,
            status: NodeStatus::Stopped,
            stop_signal: None,
        }
    }

    pub fn status(&self) -> NodeStatus {
        self.status.clone()
    }

    /// Start the node process
    pub async fn start(
        &mut self,
        data_dir: &Path,
        config: &AppConfig,
        app: tauri::AppHandle,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.process.is_some() {
            return Err("Node is already running".into());
        }

        // Ensure config.toml exists
        let secret = config.node_secret.as_ref().ok_or("No node secret configured")?;
        let public_host = format!(
            "{}:{}",
            config.public_ip.as_ref().ok_or("No public IP configured")?,
            config.port
        );

        crate::config::generate_node_config(data_dir, secret, &public_host, config.port)?;

        self.status = NodeStatus::Starting;

        // Get resource directory containing embedded Python
        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| format!("Failed to get resource dir: {}", e))?
            .join("resources"); // Tauri puts bundled resources in a "resources" subdirectory

        // Paths to embedded Python components
        let python_bin = resource_dir.join("python").join("bin").join("python3");
        let python_home = resource_dir.join("python");
        let site_packages = resource_dir.join("site-packages");
        let quip_node_dir = resource_dir.join("quip-node");
        let genesis_path = resource_dir.join("genesis_block_public.json");

        // Build PYTHONPATH: site-packages + quip-node directory
        let python_path: OsString = {
            let mut path = site_packages.as_os_str().to_owned();
            path.push(":");
            path.push(quip_node_dir.as_os_str());
            path
        };

        // Build command arguments
        let config_path = data_dir.join("config.toml");
        let num_cpus = num_cpus::get();

        // Build the Python command to invoke the CLI entry point
        // The CLI doesn't have an if __name__ == "__main__" block,
        // so we need to call the entry point function directly
        let python_code = format!(
            r#"
import sys
sys.argv = [
    'quip-network-node',
    '--config', '{}',
    'cpu',
    '--num-cpus', '{}',
    '--listen', '0.0.0.0',
    '--port', '{}',
    '--genesis-config', '{}',
    '--public-host', '{}',
    '--auto-mine',
    '--peer', 'qpu-1.nodes.quip.network',
    '--peer', 'cpu-1.quip.carback.us',
    '--peer', 'gpu-1.quip.carback.us',
    '--peer', 'gpu-2.quip.carback.us',
]
from quip_cli import quip_network_node
quip_network_node()
"#,
            config_path.display(),
            num_cpus,
            config.port,
            genesis_path.display(),
            public_host,
        );

        let mut cmd = Command::new(&python_bin);

        // Set Python environment variables
        cmd.env("PYTHONHOME", &python_home)
            .env("PYTHONPATH", &python_path)
            .env("PYTHONDONTWRITEBYTECODE", "1")  // Don't create .pyc files in bundle
            .env("PYTHONUNBUFFERED", "1");        // Unbuffered output for real-time logs

        // Run Python with -c flag to execute the CLI
        cmd.arg("-c")
            .arg(&python_code)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Log the command we're about to run
        eprintln!("[NodeManager] Starting Python at: {:?}", python_bin);
        eprintln!("[NodeManager] PYTHONHOME: {:?}", python_home);
        eprintln!("[NodeManager] PYTHONPATH: {:?}", python_path);
        eprintln!("[NodeManager] Using inline -c script to invoke quip_cli");

        let mut child = cmd.spawn()?;
        eprintln!("[NodeManager] Process spawned successfully");

        // Emit immediate feedback to frontend
        let _ = app.emit("node-log", "Node process started, initializing Python...");
        let _ = app.emit("node-status", "syncing");

        // Set up log streaming
        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        self.stop_signal = Some(stop_tx);

        // Stream stdout
        if let Some(stdout) = child.stdout.take() {
            let app_handle = app.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // Emit log event to frontend
                    let _ = app_handle.emit("node-log", &line);

                    // Detect status from log output
                    if line.contains("Connected to network") || line.contains("synced") {
                        let _ = app_handle.emit("node-status", "connected");
                    } else if line.contains("Syncing") || line.contains("synchronizing") {
                        let _ = app_handle.emit("node-status", "syncing");
                    }
                }
            });
        }

        // Stream stderr (Python logs to stderr by default)
        if let Some(stderr) = child.stderr.take() {
            let app_handle = app.clone();
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // Emit to both node-log (for display) and node-error (for errors)
                    let _ = app_handle.emit("node-log", &line);

                    // Detect status from log output
                    if line.contains("Connected to network") || line.contains("started at quic://") {
                        let _ = app_handle.emit("node-status", "connected");
                    } else if line.contains("Syncing") || line.contains("synchronizing") || line.contains("retrying") {
                        let _ = app_handle.emit("node-status", "syncing");
                    } else if line.contains("ERROR") {
                        let _ = app_handle.emit("node-error", &line);
                    }
                }
            });
        }

        // Monitor process
        let app_handle = app.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = stop_rx.recv() => {
                    // Stop signal received
                }
                status = child.wait() => {
                    match status {
                        Ok(exit_status) => {
                            if exit_status.success() {
                                let _ = app_handle.emit("node-status", "stopped");
                            } else {
                                let _ = app_handle.emit("node-status", "error");
                                let _ = app_handle.emit("node-error",
                                    format!("Node exited with code: {:?}", exit_status.code()));
                            }
                        }
                        Err(e) => {
                            let _ = app_handle.emit("node-status", "error");
                            let _ = app_handle.emit("node-error", format!("Node process error: {}", e));
                        }
                    }
                }
            }
        });

        self.status = NodeStatus::Syncing;
        Ok(())
    }

    /// Stop the node process
    pub async fn stop(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(stop_tx) = self.stop_signal.take() {
            let _ = stop_tx.send(()).await;
        }

        if let Some(mut process) = self.process.take() {
            process.kill().await?;
        }

        self.status = NodeStatus::Stopped;
        Ok(())
    }
}

impl Default for NodeManager {
    fn default() -> Self {
        Self::new()
    }
}
