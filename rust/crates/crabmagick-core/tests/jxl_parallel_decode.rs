//! Integration test: the Rayon-parallel JXL group decode must be deterministic.
//!
//! Runs only the public API, so it is unaffected by the (pre-existing) broken vendored
//! `#[cfg(test)]` unit tests. Decodes the same JXL repeatedly through the multithreaded
//! pipeline and asserts byte-for-byte identical output — a data race in the parallel group
//! decode would almost certainly produce differing pixels across runs.

use std::path::PathBuf;

use crabmagick_core::pipeline;

fn find_sample_jxl() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CRABMAGICK_TEST_JXL") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let root = PathBuf::from("/home/mattia/Work/IIIF_Server/var/storage");
    let mut stack = vec![root];
    let mut best: Option<(u64, PathBuf)> = None;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("jxl") {
                let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
                if len > 1000 && best.as_ref().map(|(b, _)| len > *b).unwrap_or(true) {
                    best = Some((len, path));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

#[test]
fn jxl_parallel_decode_is_deterministic() {
    let Some(path) = find_sample_jxl() else {
        eprintln!("SKIP: no .jxl sample found under storage root or $CRABMAGICK_TEST_JXL");
        return;
    };
    let path = path.to_str().expect("utf-8 path");

    let first = pipeline::decode_jxl(path).expect("first JXL decode");
    for run in 1..4 {
        let again = pipeline::decode_jxl(path).expect("repeat JXL decode");
        assert_eq!(
            (first.width, first.height),
            (again.width, again.height),
            "dimension mismatch on run {run}"
        );
        assert!(
            first.pixels == again.pixels,
            "parallel JXL decode not deterministic on run {run} for {path}"
        );
    }
    eprintln!(
        "OK jxl_parallel_decode_is_deterministic {}x{} ({} bytes RGB) {}",
        first.width,
        first.height,
        first.pixels.len(),
        path
    );
}
