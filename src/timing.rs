use std::time::Duration;

pub fn print_elapsed(previous: Duration, now: Duration, message: &str) -> Duration {
    println!("{} ({:.1?})", message, now - previous);
    now
}