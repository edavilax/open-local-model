//! Read-only memory maps backed by throwaway files.
//!
//! Tests for mapped tensor storage need real `memmap2::Mmap` values, and a
//! mapping needs a file. These helpers write the bytes a test wants to a
//! private temp file, map it, and unlink it straight away, so no test leaves
//! anything behind and no fixture has to exist for every small case.
//!
//! Like the rest of this crate, they panic instead of returning errors.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::{Mmap, MmapOptions};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// A path in the OS temp dir that nothing else is using. The process id
/// separates test binaries; the counter separates threads within one.
fn unique_temp_path() -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("olm-testutil-{}-{id}.bin", std::process::id()))
}

/// Writes `bytes` to a fresh temp file, maps everything from `offset` to the
/// end, then unlinks the file. The mapping keeps the pages alive.
///
/// Returns the path as well so this module's tests can check it is gone.
#[track_caller]
fn map_temp(bytes: &[u8], offset: usize) -> (Mmap, PathBuf) {
    assert!(
        offset < bytes.len(),
        "offset {offset} leaves nothing to map from {} bytes",
        bytes.len()
    );
    let path = unique_temp_path();
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    file.write_all(bytes)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));

    // SAFETY: the file was created by this call under a unique name and is
    // unlinked below, so nothing can truncate or rewrite it while mapped.
    let map = unsafe {
        MmapOptions::new()
            .offset(offset as u64)
            .len(bytes.len() - offset)
            .map(&file)
    }
    .unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));

    fs::remove_file(&path).unwrap_or_else(|e| panic!("remove {}: {e}", path.display()));
    (map, path)
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Maps `bytes` read-only. The mapping is page-aligned, like any whole-file
/// mapping of a tensor file.
#[track_caller]
pub fn mmap_bytes(bytes: &[u8]) -> Mmap {
    map_temp(bytes, 0).0
}

/// Maps `values` encoded as little-endian f32, which is how OLM stores them.
#[track_caller]
pub fn mmap_f32(values: &[f32]) -> Mmap {
    mmap_bytes(&f32_le_bytes(values))
}

/// Same contents as [`mmap_f32`], but the mapping starts one byte into its
/// page, so its address is not aligned for f32.
///
/// memmap2 accepts offsets that are not page multiples: it maps from the
/// page boundary below and hands back a pointer advanced by the remainder.
/// That is the only way to get a misaligned mapping, and it is what a
/// packed multi-tensor file with a careless offset would produce.
#[track_caller]
pub fn mmap_f32_misaligned(values: &[f32]) -> Mmap {
    let mut bytes = vec![0xAAu8]; // one pad byte, then the payload
    bytes.extend(f32_le_bytes(values));
    let map = map_temp(&bytes, 1).0;
    assert_ne!(
        map.as_ptr() as usize % align_of::<f32>(),
        0,
        "expected a misaligned mapping"
    );
    map
}

/// Maps an existing file read-only, e.g. a tensor file from `fixtures/`.
#[track_caller]
pub fn mmap_file(path: impl AsRef<Path>) -> Mmap {
    let path = path.as_ref();
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    // SAFETY: fixtures are checked-in files that nothing modifies during a
    // test run.
    unsafe { Mmap::map(&file) }.unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    #[test]
    fn mmap_bytes_round_trips() {
        let map = mmap_bytes(&[1, 2, 3, 4, 5]);
        assert_eq!(&map[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn mmap_f32_is_little_endian() {
        let map = mmap_f32(&[1.0, -2.0]);
        // 1.0 = 0x3F80_0000, -2.0 = 0xC000_0000, least significant byte first.
        assert_eq!(&map[..], &[0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0xC0]);
    }

    #[test]
    fn whole_file_mappings_are_aligned_for_f32() {
        let map = mmap_f32(&[1.0, 2.0, 3.0]);
        assert_eq!(map.as_ptr() as usize % align_of::<f32>(), 0);
    }

    #[test]
    fn misaligned_mapping_skips_the_pad_byte() {
        let map = mmap_f32_misaligned(&[1.0, -2.0]);
        assert_eq!(map.len(), 8);
        assert_eq!(&map[..], &[0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0xC0]);
        assert_ne!(map.as_ptr() as usize % align_of::<f32>(), 0);
    }

    /// The contents must outlive the directory entry, and the entry must
    /// actually be gone.
    #[test]
    fn temp_file_is_unlinked_but_mapping_stays_readable() {
        let (map, path) = map_temp(&[9, 8, 7], 0);
        assert!(!path.exists(), "{} was left behind", path.display());
        assert_eq!(&map[..], &[9, 8, 7]);
    }

    #[test]
    fn temp_paths_are_unique() {
        assert_ne!(unique_temp_path(), unique_temp_path());
    }

    #[test]
    #[should_panic(expected = "leaves nothing to map")]
    fn map_temp_rejects_offset_past_the_data() {
        map_temp(&[1, 2], 2);
    }

    #[test]
    fn mmap_file_maps_a_fixture() {
        // final_norm is [64] f32 in the toy model.
        let map = mmap_file(fixture("toy-f32/model/weights/final_norm.bin"));
        assert_eq!(map.len(), 64 * 4);
    }

    #[test]
    #[should_panic(expected = "open ")]
    fn mmap_file_names_missing_path() {
        mmap_file(fixture("nope/missing.bin"));
    }
}
