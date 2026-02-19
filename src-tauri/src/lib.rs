//! Quip Network Node Runner - Tauri Backend
//!
//! Handles node process management, configuration, and IPC with the frontend.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::menu::{Menu, MenuItem, Submenu};
use tauri::{Emitter, Manager};
use tokio::sync::Mutex;

mod node_manager;
mod config;
mod commands;

pub use node_manager::NodeManager;
pub use config::AppConfig;

/// Node connection status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum NodeStatus {
    Stopped,
    Starting,
    Syncing,
    Connected,
    Disconnected,
    Error,
}

/// Application state shared across commands
pub struct AppState {
    pub node_manager: Arc<Mutex<NodeManager>>,
    pub config: Arc<Mutex<AppConfig>>,
    pub data_dir: PathBuf,
}

impl AppState {
    pub fn new() -> Self {
        let data_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("quip-data");

        Self {
            node_manager: Arc::new(Mutex::new(NodeManager::new())),
            config: Arc::new(Mutex::new(AppConfig::default())),
            data_dir,
        }
    }
}

/// Library entry point for Tauri
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new();

    // Ensure data directory exists
    std::fs::create_dir_all(&state.data_dir).ok();

    // Load existing config if present
    if let Ok(config) = AppConfig::load(&state.data_dir) {
        *state.config.blocking_lock() = config;
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::get_node_secret,
            commands::generate_node_secret,
            commands::get_node_status,
            commands::start_node,
            commands::stop_node,
            commands::get_settings,
            commands::update_settings,
            commands::detect_public_ip,
            commands::check_port_forwarding,
        ])
        .setup(|app| {
            // Create Debug menu
            let debug_menu = Submenu::with_items(
                app,
                "Debug",
                true,
                &[
                    &MenuItem::with_id(app, "devtools", "Open DevTools", true, Some("CmdOrCtrl+Shift+I"))?,
                    &MenuItem::with_id(app, "test_ip", "Test IP Detection", true, None::<&str>)?,
                    &MenuItem::with_id(app, "show_paths", "Show Resource Paths", true, None::<&str>)?,
                ],
            )?;

            // Build full menu bar
            let menu = Menu::with_items(
                app,
                &[
                    &Submenu::with_items(
                        app,
                        "Quip Network Node Runner",
                        true,
                        &[
                            &MenuItem::with_id(app, "about", "About Quip Network Node Runner", true, None::<&str>)?,
                            &MenuItem::with_id(app, "quit", "Quit", true, Some("CmdOrCtrl+Q"))?,
                        ],
                    )?,
                    &debug_menu,
                ],
            )?;

            app.set_menu(menu)?;

            // Handle menu events
            app.on_menu_event(move |app_handle, event| {
                match event.id().as_ref() {
                    "devtools" => {
                        if let Some(window) = app_handle.get_webview_window("main") {
                            window.open_devtools();
                        }
                    }
                    "test_ip" => {
                        let app = app_handle.clone();
                        tauri::async_runtime::spawn(async move {
                            match commands::detect_public_ip().await {
                                Ok(ip) => {
                                    let _ = app.emit("debug-message", format!("Public IP detected: {}", ip));
                                    eprintln!("[Debug Menu] Public IP: {}", ip);
                                }
                                Err(e) => {
                                    let _ = app.emit("debug-message", format!("IP detection failed: {}", e));
                                    eprintln!("[Debug Menu] IP detection error: {}", e);
                                }
                            }
                        });
                    }
                    "show_paths" => {
                        if let Ok(resource_dir) = app_handle.path().resource_dir() {
                            let paths = format!(
                                "Resource dir: {:?}\nPython: {:?}\nSite-packages: {:?}",
                                resource_dir,
                                resource_dir.join("resources/python/bin/python3"),
                                resource_dir.join("resources/site-packages")
                            );
                            let _ = app_handle.emit("debug-message", paths.clone());
                            eprintln!("[Debug Menu] {}", paths);
                        }
                    }
                    "quit" => {
                        std::process::exit(0);
                    }
                    _ => {}
                }
            });

            // Open devtools in debug builds
            #[cfg(debug_assertions)]
            {
                if let Some(window) = app.get_webview_window("main") {
                    window.open_devtools();
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
