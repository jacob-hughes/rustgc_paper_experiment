use std::marker::PhantomData;
use std::rc::Rc;

struct Tree {
    l: Option<Rc<Tree>>,
    r: Option<Rc<Tree>>,
}

fn item_check(t: Option<Rc<Tree>>) -> i32 {
    match t {
        None => 0,
        Some(tree) => 1 + item_check(tree.l.clone()) + item_check(tree.r.clone()),
    }
}

fn bottom_up_tree(depth: i32) -> Option<Rc<Tree>> {
    if depth >= 0 {
        let t: Rc<Tree> = Rc::new(Tree {
            l: bottom_up_tree(depth - 1),
            r: bottom_up_tree(depth - 1),
        });
        Some(t)
    } else {
        None
    }
}

fn inner(depth: i32, iterations: i32) -> String {
    let mut chk = 0;
    for i in 1..iterations + 1 {
        // let arena = Arena::new();
        let a = bottom_up_tree(depth);
        chk += item_check(a);
    }
    format!("{}\t trees of depth {}\t check: {}", iterations, depth, chk)
}

fn main() {
    let n = std::env::args_os()
        .nth(1)
        .and_then(|s| s.into_string().ok())
        .and_then(|n| n.parse().ok())
        .unwrap_or(10);
    let min_depth = 4;
    let max_depth = if min_depth + 2 > n { min_depth + 2 } else { n };

    {
        // let arena = Arena::new();
        let depth = max_depth + 1;
        let tree = bottom_up_tree(depth);

        println!(
            "stretch tree of depth {}\t check: {}",
            depth,
            item_check(tree)
        );
    }

    // let long_lived_arena = Arena::new();
    let long_lived_tree = bottom_up_tree(max_depth);

    let messages = (min_depth..max_depth + 1)
        .filter(|&d| d % 2 == 0)
        .map(|depth| {
            let iterations = 1 << ((max_depth - depth + min_depth) as u32);
            inner(depth, iterations)
        })
        .collect::<Vec<_>>();

    for message in messages.into_iter() {
        println!("{}", message);
    }

    println!(
        "long lived tree of depth {}\t check: {}",
        max_depth,
        item_check(long_lived_tree)
    );
}
