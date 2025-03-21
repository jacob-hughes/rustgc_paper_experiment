use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

fn collect_java_sources(dir: &Path) -> Vec<String> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|e| e.to_str())
                .map_or(false, |e| e == "java")
        })
        .map(|e| e.path().display().to_string())
        .collect()
}

fn gen_syntax_errs(file: &Path) -> io::Result<()> {
    let contents = fs::read_to_string(file)?;
    let mut lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();

    let seed = 123456789;
    let mut rng = StdRng::seed_from_u64(seed);

    for line in lines.iter_mut() {
        if rng.random_bool(0.2) {
            match rng.random_range(1..=6) {
                1 => {
                    // Incorrect capitalization
                    if line.contains("if") {
                        *line = line.replace("if", "If");
                    } else if line.contains("else") {
                        *line = line.replace("else", "Else");
                    }
                }
                2 => {
                    // Remove or add parentheses
                    if line.contains("(") {
                        *line = line.replace("(", "");
                    } else if line.contains(")") {
                        *line = line.replace(")", "");
                    } else {
                        *line = format!("({})", line);
                    }
                }
                3 => {
                    // Split a string across lines
                    if line.contains("\"") {
                        let parts: Vec<&str> = line.split('"').collect();
                        if parts.len() > 1 {
                            let middle = parts[1].len() / 2;
                            let first_half = &parts[1][..middle];
                            let second_half = &parts[1][middle..];
                            *line = format!("\"{}\" +\n\"{}\"", first_half, second_half);
                        }
                    }
                }
                4 => {
                    // Missing semicolon
                    if line.ends_with(";") {
                        *line = line.trim_end_matches(';').to_string();
                    }
                }
                5 => {
                    // Misspelled keyword
                    if line.contains("System") {
                        *line = line.replace("System", "system");
                    } else if line.contains("public") {
                        *line = line.replace("public", "pub");
                    }
                }
                6 => {
                    // Mismatched quotes
                    if line.contains("\"") && line.contains("'") {
                        *line = line.replace("'", "\"");
                    } else if line.contains("'") {
                        *line = line.replace("'", "\"");
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    let modified: String = lines.join("\n");
    fs::write(file, modified)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("No Java project provided");
        std::process::exit(1);
    }

    let path = PathBuf::from(&args[1]);
    let files = collect_java_sources(&path);
    for file in files.iter().take(1000) {
        let _ = gen_syntax_errs(Path::new(file));
    }
}
