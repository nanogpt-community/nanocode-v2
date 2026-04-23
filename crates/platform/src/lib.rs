use std::env;
use std::fs::{self, File, OpenOptions, Permissions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

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

/// Atomically write bytes to `path` by writing a sibling temporary file and
/// renaming it into place.
///
/// This avoids leaving truncated JSON/session/config files behind if Pebble is
/// interrupted midway through a write. The helper intentionally creates the
/// temporary file in the destination directory so the final rename remains on
/// the same filesystem. When replacing an existing file, it follows a final
/// symlink like `fs::write` does and preserves the target file permissions.
pub fn write_atomic(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let path = resolve_existing_destination(path.as_ref())?;
    let existing_permissions = existing_writable_file_permissions(&path)?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let temp_path = unique_temp_path(&path);
    let write_result =
        write_temp_file(&temp_path, contents.as_ref(), existing_permissions.as_ref()).and_then(
            |()| {
                fs::rename(&temp_path, &path)?;
                sync_parent_dir(&path);
                Ok(())
            },
        );

    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }

    write_result
}

fn resolve_existing_destination(path: &Path) -> io::Result<PathBuf> {
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        path.canonicalize()
    } else {
        Ok(path.to_path_buf())
    }
}

fn existing_writable_file_permissions(path: &Path) -> io::Result<Option<Permissions>> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => {
            OpenOptions::new().write(true).open(path)?;
            Ok(Some(metadata.permissions()))
        }
        Ok(_) => Ok(None),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn write_temp_file(
    path: &Path,
    contents: &[u8],
    permissions: Option<&Permissions>,
) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(contents)?;
    if let Some(permissions) = permissions {
        file.set_permissions(permissions.clone())?;
    }
    file.sync_all()?;
    Ok(())
}

fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|directory| directory.sync_all());
    }
}

fn unique_temp_path(path: &Path) -> PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = process::id();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("pebble-write");
    path.with_file_name(format!(".{file_name}.{pid}.{timestamp:x}.{counter:x}.tmp"))
}

fn windows_home_from_drive_path() -> Option<PathBuf> {
    let drive = env::var_os("HOMEDRIVE").filter(|value| !value.is_empty())?;
    let path = env::var_os("HOMEPATH").filter(|value| !value.is_empty())?;

    let mut home = drive;
    home.push(path);
    Some(PathBuf::from(home))
}

#[cfg(test)]
mod tests {
    use super::{pebble_config_home, pebble_config_home_or_default, user_home_dir, write_atomic};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("pebble-platform-{name}-{nanos}"))
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

    #[test]
    fn atomic_write_creates_parent_dirs_and_replaces_file() {
        let root = temp_path("atomic-create-replace");
        let path = root.join("nested").join("state.json");

        write_atomic(&path, br#"{"value":1}"#).expect("first write succeeds");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read first write"),
            r#"{"value":1}"#
        );

        write_atomic(&path, br#"{"value":2}"#).expect("replacement write succeeds");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read replacement"),
            r#"{"value":2}"#
        );

        std::fs::remove_dir_all(root).expect("remove temp root");
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_path("atomic-permissions");
        std::fs::create_dir_all(&root).expect("create temp root");
        let path = root.join("state.json");
        std::fs::write(&path, "old").expect("write fixture");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640))
            .expect("set fixture permissions");

        write_atomic(&path, "new").expect("replacement write succeeds");

        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640);
        assert_eq!(std::fs::read_to_string(&path).expect("read file"), "new");

        std::fs::remove_dir_all(root).expect("remove temp root");
    }

    fn restore_var(name: &str, value: Option<OsString>) {
        if let Some(value) = value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }
}
