//! Tauri command handlers

use crate::{config, AppConfig, AppState, NodeStatus};

/// Get the node secret from config
#[tauri::command]
pub async fn get_node_secret(state: tauri::State<'_, AppState>) -> Result<Option<String>, String> {
    let config = state.config.lock().await;
    Ok(config.node_secret.clone())
}

/// Generate a new node secret
#[tauri::command]
pub async fn generate_node_secret(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let mut config = state.config.lock().await;
    let secret = config::generate_secret();
    config.node_secret = Some(secret.clone());
    config.save(&state.data_dir).map_err(|e| e.to_string())?;
    Ok(secret)
}

/// Get current node status
#[tauri::command]
pub async fn get_node_status(state: tauri::State<'_, AppState>) -> Result<NodeStatus, String> {
    let manager = state.node_manager.lock().await;
    Ok(manager.status())
}

/// Start the node
#[tauri::command]
pub async fn start_node(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    let mut manager = state.node_manager.lock().await;
    let config = state.config.lock().await;
    manager
        .start(&state.data_dir, &config, app)
        .await
        .map_err(|e| e.to_string())
}

/// Stop the node
#[tauri::command]
pub async fn stop_node(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut manager = state.node_manager.lock().await;
    manager.stop().await.map_err(|e| e.to_string())
}

/// Get app settings
#[tauri::command]
pub async fn get_settings(state: tauri::State<'_, AppState>) -> Result<AppConfig, String> {
    let config = state.config.lock().await;
    Ok(config.clone())
}

/// Update app settings
#[tauri::command]
pub async fn update_settings(
    state: tauri::State<'_, AppState>,
    settings: AppConfig,
) -> Result<(), String> {
    let mut config = state.config.lock().await;
    *config = settings;
    config.save(&state.data_dir).map_err(|e| e.to_string())
}

/// Detect public IP address
#[tauri::command]
pub async fn detect_public_ip() -> Result<String, String> {
    eprintln!("[DEBUG] detect_public_ip called");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| {
            eprintln!("[DEBUG] Failed to build client: {}", e);
            e.to_string()
        })?;

    eprintln!("[DEBUG] Making request to api.ipify.org...");

    let response = client
        .get("https://api.ipify.org")
        .send()
        .await
        .map_err(|e| {
            eprintln!("[DEBUG] Request failed: {}", e);
            e.to_string()
        })?
        .text()
        .await
        .map_err(|e| {
            eprintln!("[DEBUG] Failed to get text: {}", e);
            e.to_string()
        })?;

    eprintln!("[DEBUG] Got IP: {}", response.trim());
    Ok(response.trim().to_string())
}

/// Check if a port is open/forwarded using an external service
#[tauri::command]
pub async fn check_port_forwarding(ip: String, port: u16) -> Result<bool, String> {
    eprintln!("[DEBUG] check_port_forwarding called for {}:{}", ip, port);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    // Use ifconfig.co port check API
    let url = format!("https://ifconfig.co/port/{}", port);

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| {
            eprintln!("[DEBUG] Port check request failed: {}", e);
            e.to_string()
        })?
        .text()
        .await
        .map_err(|e| e.to_string())?;

    eprintln!("[DEBUG] Port check response: {}", response);

    // Parse response - ifconfig.co returns JSON like {"ip":"...","port":20049,"reachable":true/false}
    let is_open = response.to_lowercase().contains("\"reachable\":true") ||
                  response.to_lowercase().contains("\"reachable\": true");

    Ok(is_open)
}
