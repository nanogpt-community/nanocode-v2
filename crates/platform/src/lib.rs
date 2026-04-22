use std::env;
use std::path::PathBuf;

#[must_use]
pub fn user_home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("USERPROFILE")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .or_else(windows_home_from_drive_path)
}

#[must_use]
pub fn pebble_config_home() -> Option<PathBuf> {
    env::var_os("PEBBLE_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| user_home_dir().map(|home| home.join(".pebble")))
}

#[must_use]
pub fn pebble_config_home_or_default() -> PathBuf {
    pebble_config_home().unwrap_or_else(|| PathBuf::from(".pebble"))
}

#[cfg(test)]
mod tests {
    use super::{pebble_config_home, pebble_config_home_or_default, user_home_dir};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn prefers_explicit_config_home() {
        let _guard = env_lock();
        let previous_config = std::env::var_os("PEBBLE_CONFIG_HOME");
        let previous_home = std::env::var_os("HOME");
        let previous_userprofile = std::env::var_os("USERPROFILE");

        std::env::set_var("PEBBLE_CONFIG_HOME", "/tmp/pebble-config");
        std::env::set_var("HOME", "/tmp/home");
        std::env::set_var("USERPROFILE", "C:\\Users\\pebble");

        assert_eq!(
            pebble_config_home().expect("config home"),
            PathBuf::from("/tmp/pebble-config")
        );

        restore_var("PEBBLE_CONFIG_HOME", previous_config);
        restore_var("HOME", previous_home);
        restore_var("USERPROFILE", previous_userprofile);
    }

    #[test]
    fn falls_back_to_userprofile() {
        let _guard = env_lock();
        let previous_config = std::env::var_os("PEBBLE_CONFIG_HOME");
        let previous_home = std::env::var_os("HOME");
        let previous_userprofile = std::env::var_os("USERPROFILE");
        let previous_homedrive = std::env::var_os("HOMEDRIVE");
        let previous_homepath = std::env::var_os("HOMEPATH");

        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::env::remove_var("HOME");
        std::env::set_var("USERPROFILE", "C:\\Users\\pebble");
        std::env::remove_var("HOMEDRIVE");
        std::env::remove_var("HOMEPATH");

        assert_eq!(
            user_home_dir().expect("user home"),
            PathBuf::from("C:\\Users\\pebble")
        );
        assert_eq!(
            pebble_config_home_or_default(),
            PathBuf::from("C:\\Users\\pebble").join(".pebble")
        );

        restore_var("PEBBLE_CONFIG_HOME", previous_config);
        restore_var("HOME", previous_home);
        restore_var("USERPROFILE", previous_userprofile);
        restore_var("HOMEDRIVE", previous_homedrive);
        restore_var("HOMEPATH", previous_homepath);
    }

    #[test]
    fn falls_back_to_homedrive_and_homepath() {
        let _guard = env_lock();
        let previous_config = std::env::var_os("PEBBLE_CONFIG_HOME");
        let previous_home = std::env::var_os("HOME");
        let previous_userprofile = std::env::var_os("USERPROFILE");
        let previous_homedrive = std::env::var_os("HOMEDRIVE");
        let previous_homepath = std::env::var_os("HOMEPATH");

        std::env::remove_var("PEBBLE_CONFIG_HOME");
        std::env::remove_var("HOME");
        std::env::remove_var("USERPROFILE");
        std::env::set_var("HOMEDRIVE", "C:");
        std::env::set_var("HOMEPATH", "\\Users\\pebble");

        assert_eq!(
            user_home_dir().expect("user home"),
            PathBuf::from("C:\\Users\\pebble")
        );
        assert_eq!(
            pebble_config_home_or_default(),
            PathBuf::from("C:\\Users\\pebble").join(".pebble")
        );

        restore_var("PEBBLE_CONFIG_HOME", previous_config);
        restore_var("HOME", previous_home);
        restore_var("USERPROFILE", previous_userprofile);
        restore_var("HOMEDRIVE", previous_homedrive);
        restore_var("HOMEPATH", previous_homepath);
    }

    fn restore_var(name: &str, value: Option<OsString>) {
        if let Some(value) = value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }
}

fn windows_home_from_drive_path() -> Option<PathBuf> {
    let drive = env::var_os("HOMEDRIVE").filter(|value| !value.is_empty())?;
    let path = env::var_os("HOMEPATH").filter(|value| !value.is_empty())?;

    let mut home = drive;
    home.push(path);
    Some(PathBuf::from(home))
}
