mod app;
mod init;
mod input;
mod models;
mod proxy;
mod render;

fn main() {
    if let Err(error) = app::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
