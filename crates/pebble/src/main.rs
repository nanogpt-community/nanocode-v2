mod app;
mod eval;
mod eval_runner;
#[cfg(test)]
mod golden_tests;
mod init;
mod input;
mod mcp;
mod models;
mod proxy;
mod render;
mod report;
mod runtime_client;
mod session_store;
mod tool_render;
mod trace_view;
mod ui;

fn main() {
    if let Err(error) = app::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
