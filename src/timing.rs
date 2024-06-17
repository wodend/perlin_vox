use std::time::{Duration, Instant};

/// A timer for printing the elapsed time between events.
pub struct Timer {
    now: Instant,
    previous: Duration,
}
impl Timer {
    /// Create a new timer.
    pub fn new() -> Self {
        let now = Instant::now();
        let previous = now.elapsed();
        Timer { now, previous }
    }

    /// Print the elapsed time between events.
    pub fn print_elapsed(&mut self, message: &str) {
        let t = self.now.elapsed();
        println!("{} ({:.1?})", message, t - self.previous);
        self.previous = t;
    }
}
