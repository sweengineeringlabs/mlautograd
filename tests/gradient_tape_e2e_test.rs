//! Tests for GradientTape (src/api/gradient/types/gradient_tape.rs).
//!
//! @covers: GradientTape

use mlautograd::GradientTape;

#[test]
fn test_struct_gradient_tape_is_unit_struct() {
    // GradientTape is a marker/handle type — verify it can be constructed.
    let _tape = GradientTape;
}

#[test]
fn test_struct_gradient_tape_default_constructible() {
    // GradientTape derives no methods itself — the tape operations are on TapeContext.
    // This test verifies the public type exists and has no constructor panic.
    let _tape = GradientTape;
    // Integration with TapeContext is tested in mlautograd_int_test.rs
}
