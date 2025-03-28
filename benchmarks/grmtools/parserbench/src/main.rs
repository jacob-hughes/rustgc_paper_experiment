#![allow(clippy::unnecessary_wraps)]
#![feature(gc)]

use std::io::{self, BufRead, Write};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
};

use cfgrammar::Span;
use lrlex::{lrlex_mod, DefaultLexerTypes};
use lrpar::{lrpar_mod, NonStreamingLexer};
use walkdir::WalkDir;

lrlex_mod!("java.l");
lrpar_mod!("java.y");

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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("No Java project provided");
        std::process::exit(1);
    }

    let path = PathBuf::from(&args[1]);
    let files = collect_java_sources(&path);
    for (i, file) in files.iter().enumerate().take(1000) {
        let bytes =
            fs::read(file).unwrap_or_else(|_| panic!("Can't read {}.", path.to_str().unwrap()));
        let txt = String::from_utf8_lossy(&bytes);
        let lexerdef = java_l::lexerdef();
        let lexer = lexerdef.lexer(&txt);
        let (res, errs) = java_y::parse(&lexer);
        println!("Parsed {}/{}", i, files.len())
    }
}
