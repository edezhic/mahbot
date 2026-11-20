//! One small JSON map at the storage root, held under a whole-file lock.
//!
//! The lock is blocking and is taken off the async worker wherever there is one, so another
//! process's long hold cannot stall the runtime. [`LockedJson::open`] fails open (an absent,
//! unreadable or unparsable file is an empty value): these records only decide when a session's tabs
//! are closed and which profile a call carries, so a broken one must not take the surface down.
//! [`LockedJson::read_existing`] instead reports an existing record it could not read, so a caller
//! whose correctness depends on it (the profile pin) can refuse: a session's Chrome profile must
//! never be guessed.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::warn;

/// A JSON value held under its own whole-file lock; the lock is held until the value drops.
pub(crate) struct LockedJson<T> {
    root: PathBuf,
    file_name: &'static str,
    pub(crate) value: T,
    /// `true` when the record existed but could not be read or parsed, so `value` is a default:
    /// writing it back would drop what the file held.
    pub(crate) broken: bool,
    _lock: File,
}

impl<T> LockedJson<T>
where
    T: Serialize + DeserializeOwned + Default,
{
    /// Take the record under its whole-file lock, creating the lock file when needed.
    pub(crate) fn open(file_name: &'static str, lock_name: &'static str) -> Option<Self> {
        crate::util::with_block_in_place(|| {
            let root = crate::config::storage_root()?;
            let lock = lock_root(&root, lock_name)?;
            let (value, broken) = read(&root, file_name);
            Some(Self {
                root,
                file_name,
                value,
                broken,
                _lock: lock,
            })
        })
    }

    /// Like [`Self::open`], but `None` when the record does not exist: a caller that has never
    /// written one must not create the file (nor its lock) just to learn it is empty.
    pub(crate) fn open_existing(file_name: &'static str, lock_name: &'static str) -> Option<Self> {
        let root = crate::config::storage_root()?;
        if !root.join(file_name).exists() {
            return None;
        }
        Self::open(file_name, lock_name)
    }

    /// The record as written, read WITHOUT the lock: the writer replaces the file by rename
    /// ([`crate::util::write_json_record`]), so a reader sees one whole version. `Ok(None)` when no
    /// record exists; `Err(())` when one exists but cannot be read or parsed — the caller decides
    /// whether that is a refusal or an empty map. Use this for a read-only caller: it neither
    /// creates the lock file nor waits for a writer.
    pub(crate) fn read_existing(file_name: &'static str) -> Result<Option<T>, ()> {
        crate::util::with_block_in_place(|| {
            let Some(root) = crate::config::storage_root() else {
                return Ok(None);
            };
            if !root.join(file_name).exists() {
                return Ok(None);
            }
            let (value, broken) = read(&root, file_name);
            if broken { Err(()) } else { Ok(Some(value)) }
        })
    }

    /// Write the value back under the held lock. `false` when the record could not be written —
    /// the caller decides whether that is a refusal or a diagnostic.
    pub(crate) fn persist(&self) -> bool {
        let json = match serde_json::to_string(&self.value) {
            Ok(json) => json,
            Err(error) => {
                warn!(file = self.file_name, error = %error, "chrome record not written");
                return false;
            }
        };
        let path = self.root.join(self.file_name);
        match crate::util::with_block_in_place(|| crate::util::write_json_record(&path, &json)) {
            Ok(()) => true,
            Err(error) => {
                warn!(file = self.file_name, error = %error, "chrome record not written");
                false
            }
        }
    }
}

fn lock_root(root: &Path, lock_name: &str) -> Option<File> {
    if let Err(error) = std::fs::create_dir_all(root) {
        warn!(error = %error, "chrome record unavailable: storage root not creatable");
        return None;
    }
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(lock_name))
    {
        Ok(file) => file,
        Err(error) => {
            warn!(error = %error, "chrome record unavailable: lock file not openable");
            return None;
        }
    };
    match file.lock() {
        Ok(()) => Some(file),
        Err(error) => {
            warn!(error = %error, "chrome record unavailable: lock not taken");
            None
        }
    }
}

/// The record's value and whether it had to be defaulted: `true` means the file existed but could
/// not be read or parsed.
fn read<T: DeserializeOwned + Default>(root: &Path, file_name: &str) -> (T, bool) {
    let path = root.join(file_name);
    let json = match std::fs::read_to_string(&path) {
        Ok(json) => json,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return (T::default(), false),
        Err(error) => {
            warn!(error = %error, path = %path.display(), "chrome record unreadable — starting empty");
            return (T::default(), true);
        }
    };
    match serde_json::from_str(&json) {
        Ok(value) => (value, false),
        Err(error) => {
            warn!(error = %error, path = %path.display(), "chrome record unparsable — starting empty");
            (T::default(), true)
        }
    }
}
